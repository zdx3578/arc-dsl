;; properties.rkt
#lang rosette

(require "data-structures.rkt"
 "helpers.rkt")

(provide object-size object-center object-colorcount)

;; 计算对象的大小（格子数）
(define (object-size obj)
  (set-size (Object-cells obj)))

;; 计算对象的中心位置（平均坐标）
(define (object-center obj)
  (define cells (set->list (Object-cells obj)))
  (define total-i (apply + (map (λ (cell) (first (Cell-loc cell))) cells)))
  (define total-j (apply + (map (λ (cell) (second (Cell-loc cell))) cells)))
  (define n (length cells))
  (list (/ total-i n) (/ total-j n)))

;; 计算对象的颜色统计（颜色 -> 数量）
(define (object-colorcount obj)
  (define cells (set->list (Object-cells obj)))
  (define freq (make-hash))
  (for ([cell cells])
    (define c (Cell-value cell))
    (hash-update! freq c (λ (old) (+ old 1)) 1))
  (hash->list freq))
