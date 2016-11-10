#ifndef COMPILER_H
#define COMPILER_H

#define LIKNORM_LIKELY(x) x
#define LIKNORM_UNLIKELY(x) x

// Branch prediction hints
#if defined(__has_builtin)
#if __has_builtin(__builtin_expect)
#undef LIKNORM_LIKELY
#define LIKNORM_LIKELY(x) __builtin_expect(x, 1)
#undef LIKNORM_UNLIKELY
#define LIKNORM_UNLIKELY(x) __builtin_expect(x, 0)
#endif
#endif

#endif /* end of include guard: COMPILER_H */
