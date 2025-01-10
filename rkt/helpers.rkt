;; helpers.rkt
#lang rosette

(require "data-structures.rkt")

(provide grid-height grid-width grid-ref mostcolor argmax transpose)

;; 获取网格高度
(define (grid-height grid)
  (length (Grid-rows grid)))

;; 获取网格宽度（假设网格非空）
(define (grid-width grid)
  (length (first (Grid-rows grid))))

;; 获取 (i, j) 位置的颜色值
(define (grid-ref grid loc)
  (let ([i (first loc)]
        [j (second loc)])
    (list-ref (list-ref (Grid-rows grid) i) j)))

;; 计算网格中出现次数最多的颜色（作为背景色）
(define (mostcolor grid)
  (define freq (make-hash))
  (for ([i (in-range (grid-height grid))]
        [j (in-range (grid-width grid))])
    (define c (grid-ref grid (list i j)))
    (hash-update! freq c (λ (old) (+ old 1)) 1))
  ;; 找出出现次数最多的 color
  (define-values (bg _)
    (argmax (hash->list freq) (λ (p) (second p))))
  bg)

;; 找到列表中最大值元素的函数
(define (argmax lst val-fn)
  (foldl (λ (x best)
           (if (> (val-fn x) (val-fn best))
               x
               best))
         (first lst)
         (rest lst)))

;; 转置函数
(define (transpose grid)
  (apply map list grid))
