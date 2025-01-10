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

  ;; 定义一个辅助函数，从 Cell 提取位置
  (define (cell-loc cell)
    (Cell-loc cell))

  ;; 外层循环：遍历所有坐标，提取对象
  (define (outer-loop locs visited objs)
    (cond
      [(null? locs)
       (set objs)] ; 返回对象集合
      [else
       (define loc (car locs))
       (define rest-locs (cdr locs))
       (if (set-member? visited loc)
           ;; 已经属于某个对象，跳过
           (outer-loop rest-locs visited objs)
           ;; 否则，检查是否为背景色
           (let ([col (grid-ref grid loc)])
             (if (and without-bg? (equal? col bg))
                 ;; 是背景色，标记为已访问并跳过
                 (outer-loop rest-locs (set-add visited loc) objs)
                 ;; 否则，启动 BFS 提取对象
                 (let ([obj-cells (bfs-one-object grid loc col univalued? bg diagonal?)])
                   ;; 更新已访问集合
                   (define new-visited
                     (foldl (λ (cell s)
                              (set-add s (cell-loc cell)))
                            visited
                            (set->list obj-cells)))
                   ;; 创建 Object 结构
                   (define new-object (Object obj-cells))
                   ;; 添加到对象集合并继续
                   (outer-loop rest-locs new-visited (cons new-object objs))))))]))

  ;; 执行外层循环
  (define objs (outer-loop all-locs (set) '()))
  objs)