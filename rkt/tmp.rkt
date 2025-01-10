
;; bfs.rkt
#lang rosette

(require "data-structures.rkt"
         "helpers.rkt")

(provide bfs-one-object)

;; BFS: 从 start-loc 开始，提取一个对象
;; 参数：
;; - grid: Grid
;; - start-loc: (list i j)
;; - start-color: 颜色值
;; - univalued?: Boolean
;; - bg: 背景颜色（或 #f）
;; - diagonal?: Boolean
(define (bfs-one-object grid start-loc start-color univalued? bg diagonal?)
  (define h (grid-height grid))
  (define w (grid-width grid))

  ;; 纯函数式 BFS，使用递归
  (define (loop queue visited acc)
    (cond
      [(null? queue)
       acc]
      [else
       (define cand (car queue))
       (define restq (cdr queue))
       (define cand-color (grid-ref grid cand))
       ;; 判断是否将此单元格加入 acc
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

       ;; 获取候选单元格的邻居
       (define neighs (neighbors cand h w diagonal?))

       ;; 过滤符合条件且未访问的邻居
       (define new-neighs
         (filter
          (λ (nloc)
            (and (not (set-member? visited nloc))
                 (if univalued?
                     (equal? (grid-ref grid nloc) start-color)
                     (not (equal? (grid-ref grid nloc) bg)))))
          neighs))

       ;; 更新已访问集合
       (define visited2 (foldl set-add visited new-neighs))

       ;; 更新队列
       (define queue2 (append restq new-neighs))

       ;; 递归调用
       (loop queue2 visited2 acc2)]))

  ;; 初始化已访问和队列
  (define visited0 (set start-loc))
  (define queue0 (list start-loc))
  (define acc0 (set))

  ;; 运行 BFS
  (loop queue0 visited0 acc0))