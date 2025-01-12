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
  (define h (grid-height grid))
  (define w (grid-width grid))
  ;; 如果不想每次都打印，可以把这行注释掉
  (displayln "=========bfs-one-object")
  (displayln grid)

  (define (loop queue visited acc iteration)
    ;; 每次循环都打印一些信息
    (displayln (format "第~a轮循环: queue长度=~a, visited大小=~a"
                       iteration
                       (length queue)
                       (set-count visited)))
    (cond
      [(null? queue)
       acc]  ; BFS 结束，返回结果
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

      (displayln (format "cand=~a visited? ~a" cand (set-member? visited cand)))
      (printf "cand=~v (list? ~a, vector? ~a)\n" cand (list? cand) (vector? cand))
      (for ([v (in-set visited)])
        (printf "visited item=~v (list? ~a, vector? ~a)\n" v (list? v) (vector? v)))


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

       ;; 递归调用时，iteration + 1
       (loop queue2 visited2 acc2 (add1 iteration))]))

  (define visited0 (set start-loc))         ; 初始已访问
  (define queue0 (list start-loc))         ; 待处理队列
  (define acc0 (set))                      ; 收集本对象格子的集合

  (loop queue0 visited0 acc0 1))           ; 启动 BFS，从第1次循环开始
