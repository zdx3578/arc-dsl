#lang racket

(provide grid-height
         grid-width
         grid-ref
         mostcolor
         argmax
         transpose
         set-size) ;; 新增

(require "data-structures.rkt"
         racket/set)

;; ----------------------------------------------------------------------
;; 辅助函数
;; ----------------------------------------------------------------------

(define (grid-height grid)
  (length (Grid-rows grid)))

(define (grid-width grid)
  (length (first (Grid-rows grid))))

(define (grid-ref grid loc)
  (let ([i (first loc)]
        [j (second loc)])
    (list-ref (list-ref (Grid-rows grid) i) j)))

(define (argmax lst val-fn)
  (foldl (λ (x best)
           (if (> (val-fn x) (val-fn best))
               x
               best))
         (first lst)
         (rest lst)))

(define (mostcolor grid)
  (define freq (make-hash))
  (for* ([i (in-range (grid-height grid))]
         [j (in-range (grid-width grid))])
    (define c (grid-ref grid (list i j)))
    (hash-update! freq c (λ (old) (+ old 1)) 1))
  ;; 找出出现次数最多的 color
  (define max-pair
    (argmax (hash->list freq)
            (λ (p) (cdr p)))) ; 这里使用 cdr 获取出现次数
  (define bg        (car max-pair))
  (define col-count (cdr max-pair))
  bg)


(define (transpose grid)
  (apply map list grid))

;; 新增：定义 set-size
(define (set-size s)
  (length (set->list s)))
