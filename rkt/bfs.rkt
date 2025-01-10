#lang racket

(provide bfs-one-object)

(require racket/set
         helpers
         data-structures)

;; ----------------------------------------------------------------------
;; BFS 实现
;; ----------------------------------------------------------------------

;; 根据 diagonal? 参数返回相应的邻居
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

;; 从起点 loc 开始，扩展出一个对象
;; 如果 univalued? 为 #t，则对象内颜色必须 == start-color
;; 否则，只要不是背景色 bg 就可加入
(define (bfs-one-object grid start-loc start-color univalued? bg diagonal?)
  (define h (grid-height grid))
  (define w (grid-width grid))

  ;; 使用纯函数式 BFS，使用 tail recursion + visited set + queue list
  (define (loop queue visited acc)
    (cond
      [(null? queue)
       acc]
      [else
       (define cand (car queue))
       (define restq (cdr queue))
       (define cand-color (grid-ref grid cand))
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

       ;; 找 cand 的邻居
       (define neighs (neighbors cand h w diagonal?))
       ;; 过滤：尚未访问、且符合颜色条件
       (define new-neighs
         (filter
          (λ (nloc)
            (and (not (set-member? visited nloc))
                 (if univalued?
                     (equal? (grid-ref grid nloc) start-color)
                     (not (equal? (grid-ref grid nloc) bg)))))
          neighs))
       ;; 标记 new-neighs 为已访问
       (define visited2 (foldl set-add visited new-neighs))
       ;; 入队
       (define queue2 (append restq new-neighs))
       ;; 递归调用
       (loop queue2 visited2 acc2)]))

  ;; 初始化
  (define visited0 (set start-loc))
  (define queue0 (list start-loc))
  (define acc0 (set))

  ;; 执行 BFS
  (loop queue0 visited0 acc0))
