// pd: prompt prefill on one llama-server (the GPU's), token generation on
// another (the host + Xeon Phi design A), the two overlapped across a
// stream of prompts. Each prompt's state moves between them as a slot file
// (POST /slots/0?action=save on the prefill server, ?action=restore on the
// decode server; both started with the same --slot-save-path).
//
// The prefill server gets the prompt minus its last token; the decode
// server gets the whole prompt, finds n-1 tokens cached and evaluates one.
// A hybrid (recurrent) model cannot drop tokens from its state, so this is
// the only split the decode server can reuse without reprocessing.
//
// usage: pd [options] PROMPT_FILE...      (each file is one prompt)
//   -P PORT    prefill server (default 8091)    -D PORT  decode server (8092)
//   -m MODE    pipe (default) | serial | donly  -n N     tokens to generate (128)
//   -r R       send every file R times (1)      -c       wrap as a chat user turn
//   -g         greedy (temperature 0)           -o DIR   write req-K.txt there
//   -f NAME    slot file prefix (pd)
// Sampling otherwise: temperature 0.5, top-k 75, top-p 0.95, min-p 0.05,
// repeat penalty 1.2, seed 1.
#define _GNU_SOURCE
#include <arpa/inet.h>
#include <errno.h>
#include <netinet/in.h>
#include <pthread.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <strings.h>
#include <sys/socket.h>
#include <time.h>
#include <unistd.h>

struct buf { char *p; size_t n, cap; };

static void b_put(struct buf *b, const char *s, size_t n) {
    if (b->n + n + 1 > b->cap) {
        size_t cap = b->cap ? b->cap : 4096;
        while (cap < b->n + n + 1) cap *= 2;
        char *p = realloc(b->p, cap);
        if (!p) { perror("realloc"); exit(1); }
        b->p = p; b->cap = cap;
    }
    memcpy(b->p + b->n, s, n);
    b->n += n;
    b->p[b->n] = 0;
}

#define B_LIT(b, s) b_put((b), "" s, sizeof(s) - 1)

static void b_fmt(struct buf *b, const char *fmt, ...) {
    char tmp[512];
    va_list ap;
    va_start(ap, fmt);
    int k = vsnprintf(tmp, sizeof tmp, fmt, ap);
    va_end(ap);
    if (k < 0) return;
    if ((size_t) k < sizeof tmp) { b_put(b, tmp, (size_t) k); return; }
    char *big = malloc((size_t) k + 1);
    if (!big) { perror("malloc"); exit(1); }
    va_start(ap, fmt);
    vsnprintf(big, (size_t) k + 1, fmt, ap);
    va_end(ap);
    b_put(b, big, (size_t) k);
    free(big);
}

// JSON string literal of s (quotes included)
static void b_jstr(struct buf *b, const char *s) {
    B_LIT(b, "\"");
    for (const unsigned char *c = (const unsigned char *) s; *c; c++) {
        switch (*c) {
        case '"':  B_LIT(b, "\\\""); break;
        case '\\': B_LIT(b, "\\\\"); break;
        case '\n': B_LIT(b, "\\n"); break;
        case '\r': B_LIT(b, "\\r"); break;
        case '\t': B_LIT(b, "\\t"); break;
        default:
            if (*c < 0x20) b_fmt(b, "\\u%04x", *c);
            else b_put(b, (const char *) c, 1);
        }
    }
    B_LIT(b, "\"");
}

static double now_s(void) {
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return (double) t.tv_sec + (double) t.tv_nsec * 1e-9;
}

// ---- HTTP/1.1 POST to 127.0.0.1, Connection: close ----

static int http_post(int port, const char *path, const struct buf *body, struct buf *out) {
    int fd = socket(AF_INET, SOCK_STREAM, 0);
    if (fd < 0) { perror("socket"); return -1; }
    struct sockaddr_in a = { .sin_family = AF_INET, .sin_port = htons((uint16_t) port) };
    a.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
    if (connect(fd, (struct sockaddr *) &a, sizeof a) < 0) {
        fprintf(stderr, "pd: connect :%d: %s\n", port, strerror(errno));
        close(fd);
        return -1;
    }
    struct buf req = {0};
    b_fmt(&req, "POST %s HTTP/1.1\r\nHost: 127.0.0.1:%d\r\nContent-Type: application/json\r\n"
                "Content-Length: %zu\r\nConnection: close\r\n\r\n", path, port, body->n);
    b_put(&req, body->p, body->n);
    for (size_t off = 0; off < req.n;) {
        ssize_t k = write(fd, req.p + off, req.n - off);
        if (k < 0) { if (errno == EINTR) continue; perror("write"); free(req.p); close(fd); return -1; }
        off += (size_t) k;
    }
    free(req.p);
    struct buf raw = {0};
    char chunk[65536];
    for (;;) {
        ssize_t k = read(fd, chunk, sizeof chunk);
        if (k < 0) { if (errno == EINTR) continue; perror("read"); break; }
        if (k == 0) break;
        b_put(&raw, chunk, (size_t) k);
    }
    close(fd);
    if (!raw.p) return -1;
    int status = 0;
    sscanf(raw.p, "HTTP/%*s %d", &status);
    char *hend = strstr(raw.p, "\r\n\r\n");
    if (!hend) { free(raw.p); return -1; }
    *hend = 0;
    int chunked = strcasestr(raw.p, "transfer-encoding: chunked") != NULL;
    char *p = hend + 4, *end = raw.p + raw.n;
    out->n = 0;
    if (!chunked) {
        b_put(out, p, (size_t) (end - p));
    } else {
        while (p < end) {
            char *e;
            unsigned long len = strtoul(p, &e, 16);
            char *crlf = strstr(e, "\r\n");
            if (!crlf || len == 0) break;
            p = crlf + 2;
            if (p + len > end) len = (unsigned long) (end - p);
            b_put(out, p, len);
            p += len + 2;
        }
    }
    if (!out->p) B_LIT(out, "");
    free(raw.p);
    return status;
}

// ---- minimal JSON reader: values located by a dotted path of object keys ----

static const char *j_ws(const char *s) { while (*s == ' ' || *s == '\n' || *s == '\r' || *s == '\t') s++; return s; }

static const char *j_skip(const char *s) {
    s = j_ws(s);
    if (*s == '"') {
        for (s++; *s && *s != '"'; s++) if (*s == '\\' && s[1]) s++;
        return *s ? s + 1 : NULL;
    }
    if (*s == '{' || *s == '[') {
        int depth = 0;
        for (; *s; s++) {
            if (*s == '"') { s = j_skip(s); if (!s) return NULL; s--; continue; }
            if (*s == '{' || *s == '[') depth++;
            else if (*s == '}' || *s == ']') { if (--depth == 0) return s + 1; }
        }
        return NULL;
    }
    while (*s && *s != ',' && *s != '}' && *s != ']' && *s != ' ' && *s != '\n') s++;
    return s;
}

// value of key in the object at obj ('{'), or NULL
static const char *j_key(const char *obj, const char *key, size_t klen) {
    obj = j_ws(obj);
    if (*obj != '{') return NULL;
    const char *s = obj + 1;
    for (;;) {
        s = j_ws(s);
        if (*s != '"') return NULL;
        const char *k0 = s + 1;
        const char *k1 = j_skip(s);
        if (!k1) return NULL;
        s = j_ws(k1);
        if (*s != ':') return NULL;
        const char *v = j_ws(s + 1);
        if ((size_t) (k1 - 1 - k0) == klen && memcmp(k0, key, klen) == 0) return v;
        s = j_skip(v);
        if (!s) return NULL;
        s = j_ws(s);
        if (*s != ',') return NULL;
        s++;
    }
}

static const char *j_path(const char *json, const char *path) {
    const char *v = json;
    while (v && *path) {
        const char *dot = strchr(path, '.');
        size_t len = dot ? (size_t) (dot - path) : strlen(path);
        v = j_key(v, path, len);
        path += len + (dot ? 1 : 0);
    }
    return v;
}

static double j_num(const char *json, const char *path, double dflt) {
    const char *v = j_path(json, path);
    if (!v) return dflt;
    char *e;
    double d = strtod(v, &e);
    return e == v ? dflt : d;
}

static void utf8(struct buf *b, unsigned cp) {
    char o[4];
    if (cp < 0x80) { o[0] = (char) cp; b_put(b, o, 1); }
    else if (cp < 0x800) { o[0] = (char) (0xC0 | cp >> 6); o[1] = (char) (0x80 | (cp & 0x3F)); b_put(b, o, 2); }
    else if (cp < 0x10000) { o[0] = (char) (0xE0 | cp >> 12); o[1] = (char) (0x80 | (cp >> 6 & 0x3F)); o[2] = (char) (0x80 | (cp & 0x3F)); b_put(b, o, 3); }
    else { o[0] = (char) (0xF0 | cp >> 18); o[1] = (char) (0x80 | (cp >> 12 & 0x3F)); o[2] = (char) (0x80 | (cp >> 6 & 0x3F)); o[3] = (char) (0x80 | (cp & 0x3F)); b_put(b, o, 4); }
}

// unescaped copy of the string at path (malloc), or NULL
static char *j_str(const char *json, const char *path) {
    const char *s = j_path(json, path);
    if (!s || *s != '"') return NULL;
    struct buf b = {0};
    B_LIT(&b, "");
    for (s++; *s && *s != '"'; s++) {
        if (*s != '\\') { b_put(&b, s, 1); continue; }
        s++;
        switch (*s) {
        case 'n': B_LIT(&b, "\n"); break;
        case 't': B_LIT(&b, "\t"); break;
        case 'r': B_LIT(&b, "\r"); break;
        case 'b': B_LIT(&b, "\b"); break;
        case 'f': B_LIT(&b, "\f"); break;
        case 'u': {
            unsigned cp = 0;
            if (sscanf(s + 1, "%4x", &cp) != 1) break;
            s += 4;
            if (cp >= 0xD800 && cp < 0xDC00 && s[1] == '\\' && s[2] == 'u') {
                unsigned lo = 0;
                if (sscanf(s + 3, "%4x", &lo) == 1 && lo >= 0xDC00 && lo < 0xE000) {
                    cp = 0x10000 + ((cp - 0xD800) << 10) + (lo - 0xDC00);
                    s += 6;
                }
            }
            utf8(&b, cp);
        } break;
        default: b_put(&b, s, 1);
        }
    }
    return b.p;
}

// integer array at path; returns count, -1 if absent
static int j_ints(const char *json, const char *path, int **out) {
    const char *s = j_path(json, path);
    if (!s || *s != '[') return -1;
    int n = 0, cap = 1024;
    int *v = malloc(sizeof(int) * (size_t) cap);
    for (s++;;) {
        s = j_ws(s);
        if (*s == ']') break;
        char *e;
        long x = strtol(s, &e, 10);
        if (e == s) { free(v); return -1; }
        if (n == cap) { cap *= 2; v = realloc(v, sizeof(int) * (size_t) cap); }
        v[n++] = (int) x;
        s = j_ws(e);
        if (*s == ',') s++;
    }
    *out = v;
    return n;
}

// ---- the run ----

enum mode { M_PIPE, M_SERIAL, M_DONLY };

struct req {
    int id;
    const char *name;
    int *tok, ntok;
    char file[96];
    double t_pf0, t_pf1, t_sv1, t_rs0, t_rs1, t_dc1;
    double pf_n, pf_ms, sv_ms, rs_ms, dc_cache, dc_pn, dc_pms, dc_gn, dc_gms;
    char *out;
    int ok;
};

static struct {
    int pport, dport, n_predict, greedy, chat;
    enum mode mode;
    const char *prefix;
    struct req *r;
    int nr;
    double t0;
    pthread_mutex_t mu;
    pthread_cond_t cv;
    int n_saved, n_restored, failed;
} G = { .pport = 8091, .dport = 8092, .n_predict = 128, .prefix = "pd",
        .mu = PTHREAD_MUTEX_INITIALIZER, .cv = PTHREAD_COND_INITIALIZER };

static void b_tokens(struct buf *b, const int *t, int n) {
    B_LIT(b, "[");
    for (int i = 0; i < n; i++) b_fmt(b, i ? ",%d" : "%d", t[i]);
    B_LIT(b, "]");
}

static int post_ok(int port, const char *path, const struct buf *body, struct buf *out, const char *what, int id) {
    int st = http_post(port, path, body, out);
    if (st != 200) {
        fprintf(stderr, "pd: request %d %s: HTTP %d: %.400s\n", id, what, st, out->p ? out->p : "");
        return 0;
    }
    return 1;
}

static void fail(void) {
    pthread_mutex_lock(&G.mu);
    G.failed = 1;
    pthread_cond_broadcast(&G.cv);
    pthread_mutex_unlock(&G.mu);
}

// prefill request k on the prefill server and save its slot
static int prefill(struct req *q) {
    struct buf body = {0}, out = {0};
    q->t_pf0 = now_s() - G.t0;
    B_LIT(&body, "{\"prompt\":");
    b_tokens(&body, q->tok, q->ntok - 1);
    b_fmt(&body, ",\"n_predict\":0,\"id_slot\":0,\"cache_prompt\":false}");
    int ok = post_ok(G.pport, "/completion", &body, &out, "prefill", q->id);
    if (ok) {
        q->t_pf1 = now_s() - G.t0;
        q->pf_n = j_num(out.p, "timings.prompt_n", -1);
        q->pf_ms = j_num(out.p, "timings.prompt_ms", -1);
        // the file name alternates: the decode thread restores k-1 while k is saved
        pthread_mutex_lock(&G.mu);
        while (!G.failed && G.n_restored < q->id - 1) pthread_cond_wait(&G.cv, &G.mu);
        pthread_mutex_unlock(&G.mu);
        body.n = 0;
        b_fmt(&body, "{\"filename\":\"%s\"}", q->file);
        ok = !G.failed && post_ok(G.pport, "/slots/0?action=save", &body, &out, "save", q->id);
    }
    if (ok) {
        q->t_sv1 = now_s() - G.t0;
        q->sv_ms = j_num(out.p, "timings.save_ms", -1);
        double n = j_num(out.p, "n_saved", -1);
        if ((int) n != q->ntok - 1) {
            fprintf(stderr, "pd: request %d: saved %.0f tokens, expected %d\n", q->id, n, q->ntok - 1);
            ok = 0;
        }
    }
    free(body.p); free(out.p);
    return ok;
}

// restore request k on the decode server (unless donly) and generate
static int decode(struct req *q) {
    struct buf body = {0}, out = {0};
    int ok = 1;
    q->t_rs0 = now_s() - G.t0;
    if (G.mode != M_DONLY) {
        b_fmt(&body, "{\"filename\":\"%s\"}", q->file);
        ok = post_ok(G.dport, "/slots/0?action=restore", &body, &out, "restore", q->id);
        if (ok) {
            q->rs_ms = j_num(out.p, "timings.restore_ms", -1);
            double n = j_num(out.p, "n_restored", -1);
            if ((int) n != q->ntok - 1) {
                fprintf(stderr, "pd: request %d: restored %.0f tokens, expected %d\n", q->id, n, q->ntok - 1);
                ok = 0;
            }
        }
        pthread_mutex_lock(&G.mu);
        G.n_restored = q->id + 1;
        pthread_cond_broadcast(&G.cv);
        pthread_mutex_unlock(&G.mu);
    }
    q->t_rs1 = now_s() - G.t0;
    if (ok) {
        body.n = 0;
        B_LIT(&body, "{\"prompt\":");
        b_tokens(&body, q->tok, q->ntok);
        b_fmt(&body, ",\"n_predict\":%d,\"id_slot\":0,\"cache_prompt\":%s,\"seed\":1", G.n_predict,
              G.mode == M_DONLY ? "false" : "true");
        if (G.greedy) b_fmt(&body, ",\"temperature\":0}");
        else b_fmt(&body, ",\"temperature\":0.5,\"top_k\":75,\"top_p\":0.95,\"min_p\":0.05,\"repeat_penalty\":1.2}");
        ok = post_ok(G.dport, "/completion", &body, &out, "decode", q->id);
    }
    if (ok) {
        q->t_dc1 = now_s() - G.t0;
        q->dc_cache = j_num(out.p, "timings.cache_n", -1);
        q->dc_pn = j_num(out.p, "timings.prompt_n", -1);
        q->dc_pms = j_num(out.p, "timings.prompt_ms", -1);
        q->dc_gn = j_num(out.p, "timings.predicted_n", -1);
        q->dc_gms = j_num(out.p, "timings.predicted_ms", -1);
        q->out = j_str(out.p, "content");
        if (G.mode != M_DONLY && ((int) q->dc_cache != q->ntok - 1 || (int) q->dc_pn != 1)) {
            fprintf(stderr, "pd: request %d: decode server cached %.0f and evaluated %.0f tokens, expected %d and 1\n",
                    q->id, q->dc_cache, q->dc_pn, q->ntok - 1);
            ok = 0;
        }
    }
    free(body.p); free(out.p);
    return ok;
}

static void *prefill_thread(void *arg) {
    (void) arg;
    for (int k = 0; k < G.nr && !G.failed; k++) {
        if (!prefill(&G.r[k])) { fail(); break; }
        pthread_mutex_lock(&G.mu);
        G.n_saved = k + 1;
        pthread_cond_broadcast(&G.cv);
        pthread_mutex_unlock(&G.mu);
    }
    return NULL;
}

static char *slurp(const char *path) {
    FILE *f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "pd: %s: %s\n", path, strerror(errno)); exit(1); }
    struct buf b = {0};
    char tmp[65536];
    size_t k;
    while ((k = fread(tmp, 1, sizeof tmp, f)) > 0) b_put(&b, tmp, k);
    fclose(f);
    if (!b.p) B_LIT(&b, "");
    return b.p;
}

// prompt text -> token ids, through the server that will see them first
static int tokenize(const char *text, int **tok) {
    int port = G.mode == M_DONLY ? G.dport : G.pport;
    struct buf body = {0}, out = {0};
    char *prompt = NULL;
    if (G.chat) {
        B_LIT(&body, "{\"messages\":[{\"role\":\"user\",\"content\":");
        b_jstr(&body, text);
        B_LIT(&body, "}]}");
        if (!post_ok(port, "/apply-template", &body, &out, "apply-template", -1)) exit(1);
        prompt = j_str(out.p, "prompt");
        body.n = 0;
    }
    B_LIT(&body, "{\"content\":");
    b_jstr(&body, prompt ? prompt : text);
    B_LIT(&body, ",\"add_special\":true,\"parse_special\":true}");
    if (!post_ok(port, "/tokenize", &body, &out, "tokenize", -1)) exit(1);
    int n = j_ints(out.p, "tokens", tok);
    free(prompt); free(body.p); free(out.p);
    return n;
}

static double overlap(void) {
    // time during which a prefill (prefill + save) and a decode (restore + generate) both ran
    double s = 0;
    for (int i = 0; i < G.nr; i++)
        for (int j = 0; j < G.nr; j++) {
            double a = G.r[i].t_pf0 > G.r[j].t_rs0 ? G.r[i].t_pf0 : G.r[j].t_rs0;
            double b = G.r[i].t_sv1 < G.r[j].t_dc1 ? G.r[i].t_sv1 : G.r[j].t_dc1;
            if (b > a) s += b - a;
        }
    return s;
}

int main(int argc, char **argv) {
    int opt, repeat = 1;
    const char *outdir = NULL;
    while ((opt = getopt(argc, argv, "P:D:m:n:r:cgo:f:")) != -1) {
        switch (opt) {
        case 'P': G.pport = atoi(optarg); break;
        case 'D': G.dport = atoi(optarg); break;
        case 'n': G.n_predict = atoi(optarg); break;
        case 'r': repeat = atoi(optarg); break;
        case 'c': G.chat = 1; break;
        case 'g': G.greedy = 1; break;
        case 'o': outdir = optarg; break;
        case 'f': G.prefix = optarg; break;
        case 'm':
            if (!strcmp(optarg, "pipe")) G.mode = M_PIPE;
            else if (!strcmp(optarg, "serial")) G.mode = M_SERIAL;
            else if (!strcmp(optarg, "donly")) G.mode = M_DONLY;
            else { fprintf(stderr, "pd: -m pipe|serial|donly\n"); return 2; }
            break;
        default:
            fprintf(stderr, "usage: pd [-P port] [-D port] [-m pipe|serial|donly] [-n N] [-r R] [-c] [-g] [-o dir] [-f prefix] FILE...\n");
            return 2;
        }
    }
    if (optind >= argc || repeat < 1) { fprintf(stderr, "pd: no prompt files\n"); return 2; }
    int nf = argc - optind;
    G.nr = nf * repeat;
    G.r = calloc((size_t) G.nr, sizeof *G.r);
    for (int f = 0; f < nf; f++) {
        char *text = slurp(argv[optind + f]);
        int *tok = NULL;
        int n = tokenize(text, &tok);
        free(text);
        if (n < 2) { fprintf(stderr, "pd: %s: %d tokens, need at least 2\n", argv[optind + f], n); return 1; }
        for (int k = 0; k < repeat; k++) {
            struct req *q = &G.r[k * nf + f];
            q->name = argv[optind + f];
            q->tok = tok;
            q->ntok = n;
        }
    }
    for (int k = 0; k < G.nr; k++) {
        G.r[k].id = k;
        snprintf(G.r[k].file, sizeof G.r[k].file, "%s-%d-%d.bin", G.prefix, (int) getpid(), k % 2);
    }

    G.t0 = now_s();
    if (G.mode == M_PIPE) {
        pthread_t th;
        pthread_create(&th, NULL, prefill_thread, NULL);
        for (int k = 0; k < G.nr; k++) {
            pthread_mutex_lock(&G.mu);
            while (!G.failed && G.n_saved <= k) pthread_cond_wait(&G.cv, &G.mu);
            int stop = G.failed;
            pthread_mutex_unlock(&G.mu);
            if (stop) break;
            if (!(G.r[k].ok = decode(&G.r[k]))) { fail(); break; }
        }
        pthread_join(th, NULL);
    } else {
        for (int k = 0; k < G.nr; k++) {
            if (G.mode == M_SERIAL && !prefill(&G.r[k])) { G.failed = 1; break; }
            if (!(G.r[k].ok = decode(&G.r[k]))) { G.failed = 1; break; }
        }
    }
    double wall = now_s() - G.t0;

    printf("mode %s, %d request(s), n_predict %d, %s\n",
           G.mode == M_PIPE ? "pipe" : G.mode == M_SERIAL ? "serial" : "donly", G.nr, G.n_predict,
           G.greedy ? "greedy" : "temp 0.5 top-k 75 top-p 0.95 min-p 0.05 rp 1.2");
    printf("%3s %6s | %9s %9s %8s | %8s %6s %5s %9s %6s %9s | %8s %8s\n", "req", "tokens", "pf tok", "pf tok/s",
           "save ms", "rest ms", "cache", "eval", "eval ms", "gen", "gen tok/s", "start s", "end s");
    double pf_n = 0, pf_ms = 0, gn = 0, gms = 0, dpn = 0, dpms = 0;
    int done = 0;
    for (int k = 0; k < G.nr; k++) {
        struct req *q = &G.r[k];
        if (!q->ok) continue;
        done++;
        pf_n += q->pf_n > 0 ? q->pf_n : 0; pf_ms += q->pf_ms > 0 ? q->pf_ms : 0;
        gn += q->dc_gn; gms += q->dc_gms; dpn += q->dc_pn; dpms += q->dc_pms;
        printf("%3d %6d | %9.0f %9.2f %8.1f | %8.1f %6.0f %5.0f %9.1f %6.0f %9.2f | %8.2f %8.2f\n", k, q->ntok,
               q->pf_n, q->pf_ms > 0 ? q->pf_n * 1000 / q->pf_ms : 0, q->sv_ms, q->rs_ms, q->dc_cache, q->dc_pn,
               q->dc_pms, q->dc_gn, q->dc_gms > 0 ? q->dc_gn * 1000 / q->dc_gms : 0,
               G.mode == M_DONLY ? q->t_rs0 : q->t_pf0, q->t_dc1);
    }
    printf("done %d/%d in %.2f s wall\n", done, G.nr, wall);
    if (pf_ms > 0) printf("prefill server: %.0f tokens in %.1f s = %.2f tok/s\n", pf_n, pf_ms / 1000, pf_n * 1000 / pf_ms);
    if (dpms > 0) printf("decode server prompt: %.0f tokens in %.1f s = %.2f tok/s\n", dpn, dpms / 1000, dpn * 1000 / dpms);
    if (gms > 0) printf("decode server generation: %.0f tokens in %.1f s = %.2f tok/s\n", gn, gms / 1000, gn * 1000 / gms);
    if (G.mode == M_PIPE) printf("prefill and decode overlapped %.2f s of %.2f s\n", overlap(), wall);

    if (outdir) {
        for (int k = 0; k < G.nr; k++) {
            if (!G.r[k].out) continue;
            char path[4096];
            snprintf(path, sizeof path, "%s/req-%d.txt", outdir, k);
            FILE *f = fopen(path, "w");
            if (!f) { fprintf(stderr, "pd: %s: %s\n", path, strerror(errno)); continue; }
            fputs(G.r[k].out, f);
            fclose(f);
        }
    } else {
        for (int k = 0; k < G.nr; k++)
            if (G.r[k].out) printf("\n=== request %d (%s)\n%s\n", k, G.r[k].name, G.r[k].out);
    }
    return G.failed ? 1 : 0;
}
