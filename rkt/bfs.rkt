#lang racket

(provide bfs-one-object)

(require "helpers.rkt"
         "data-structures.rkt"
         racket/set)

;; ----------------------------------------------------------------------
;; 1. 定义 neighbors 函数
;; ----------------------------------------------------------------------
(define (neighbors loc h w diagonal?)
  (define i (first loc))
  (define j (second loc))
  (define base-neighs
    (list
     (list i     (add1 j))   ; 右
     (list i     (sub1 j))   ; 左
     (list (add1 i) j)       ; 下
     (list (sub1 i) j)))     ; 上
  (define diag-neighs
    (if diagonal?
        (list
         (list (add1 i) (add1 j)) ; 右下
         (list (add1 i) (sub1 j)) ; 左下
         (list (sub1 i) (add1 j)) ; 右上
         (list (sub1 i) (sub1 j))) ; 左上
        '()))
  (define all-neighs (append base-neighs diag-neighs))
  ;; 过滤越界
  (filter (λ (p)
            (define x (first p))
            (define y (second p))
            (and (>= x 0) (< x h) (>= y 0) (< y w)))
          all-neighs))

;; ----------------------------------------------------------------------
;; 2. BFS 实现：bfs-one-object
;; ----------------------------------------------------------------------
;; - 如果 univalued?=#t，则对象内颜色必须与起点一样
;; - 如果 univalued?=#f，则只要不是背景色 bg 都可加入同一对象
(define (bfs-one-object grid start-loc start-color univalued? bg diagonal?)
  ;; grid: Grid
  ;; start-loc: (i j)
  ;; start-color: color
  ;; univalued?: Boolean
  ;; bg: 背景颜色 or #f
  ;; diagonal?: Boolean
  (define h (grid-height grid))
  (define w (grid-width grid))

  ;; BFS 内部循环
  (define (loop queue visited acc)
    (cond
      [(null? queue)
       acc]  ; 返回 acc (set-of-Cell)
      [else
       ;; 取队首
       (define cand  (car queue))
       (define restq (cdr queue))

       (define cand-color (grid-ref grid cand))

       ;; 判断是否符合条件 => 加到 acc 里
       (define acc2
         (cond
           [(and (not univalued?)
                 (not (equal? cand-color bg)))
            (set-add acc (Cell cand-color cand))]
           [(and univalued?
                 (equal? cand-color start-color))
            (set-add acc (Cell cand-color cand))]
           [else
            acc]))

       ;; 找邻居
       (define neighs (neighbors cand h w diagonal?))

       ;; 过滤：尚未访问 & 符合颜色条件
       (define new-neighs
         (filter
          (λ (nloc)
            (and (not (set-member? visited nloc))
                 (if univalued?
                     (equal? (grid-ref grid nloc) start-color)
                     (not (equal? (grid-ref grid nloc) bg)))))
          neighs))

       ;; 标记访问
       (define visited2 (foldl set-add visited new-neighs))
       ;; 入队
       (define queue2 (append restq new-neighs))
       (loop queue2 visited2 acc2)]))

  ;; 初始已访问
  (define visited0 (set start-loc))
  ;; 待处理队列
  (define queue0 (list start-loc))
  ;; 收集本对象格子的集合
  (define acc0 (set))

  ;; 启动 BFS
  (loop queue0 visited0 acc0))
