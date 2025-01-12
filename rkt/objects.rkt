;; objects.rkt
#lang rosette

(require "data-structures.rkt"
         "helpers.rkt"
         "bfs.rkt")

(provide objects)

;; 提取网格中的所有对象
;; 参数：
;; - grid: Grid
;; - univalued?: Boolean
;; - diagonal?: Boolean
;; - without-bg?: Boolean
;; 返回：
;; - Set of Objects
(define (objects grid univalued? diagonal? without-bg?)
  (define h (grid-height grid))
  (define w (grid-width grid))
  (displayln h)
  (displayln w)

  ;; 确定背景颜色
  (define bg
    (if without-bg?
        (mostcolor grid)
        #f)) ; #f 表示不排除任何颜色

  ;; 获取所有坐标
  (define all-locs
    (for*/list ([i (in-range h)]
                [j (in-range w)])
      (list i j)))

  (displayln all-locs)

  ;; 定义一个辅助函数，从 Cell 提取位置
  (define (cell-loc cell)
    (Cell-loc cell))

  ;; 外层循环：遍历所有坐标，提取对象
  (define (outer-loop locs visited objs)
    (cond
      [(null? locs)
       ;; 改成把列表里的所有对象构造成一个 set
       (list->set objs)]
      [else
       (define loc (car locs))
       (define rest-locs (cdr locs))
       (if (set-member? visited loc)
           (outer-loop rest-locs visited objs)
           (let ([col (grid-ref grid loc)])
             (if (and without-bg? (equal? col bg))
                 (outer-loop rest-locs (set-add visited loc) objs)
                 (let ([obj-cells (bfs-one-object grid loc col
                                                  univalued? bg diagonal?)])
                   (define new-visited
                     (foldl (λ (cell s)
                              (set-add s (Cell-loc cell)))
                            visited
                            (set->list obj-cells)))
                   (define new-object (Object obj-cells))
                   (outer-loop rest-locs new-visited
                               (cons new-object objs))))))]))
  (outer-loop all-locs (set) '()))