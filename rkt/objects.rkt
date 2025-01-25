;; objects.rkt
#lang rosette

;;; #lang racket

;;; (require racket/set)

(require "data-structures.rkt"
          racket/set
         racket/list)  ;; 用于一些列表操作

;; ----------------------------------------------------------------------
;; 1. 定义 Grid 及辅助函数
;; ----------------------------------------------------------------------
;;; (struct Grid (rows) #:transparent)
;; 例如: (Grid '((0 0 0)
;;               (7 7 7)
;;               (0 7 0)))

;; 获取 grid 的高度和宽度
(define (grid-height grid)
  (length (Grid-rows grid)))

(define (grid-width grid)
  (length (first (Grid-rows grid))))

;; 读取网格某坐标的颜色
(define (grid-ref grid loc)
  (let ([r (first loc)]
        [c (second loc)])
    (list-ref (list-ref (Grid-rows grid) r) c)))

;; 生成所有坐标
(define (all-locs grid)
  (for*/list ([i (in-range (grid-height grid))]
              [j (in-range (grid-width grid))])
    (list i j)))


;; ----------------------------------------------------------------------
;; 2. 计算“网格中出现最多”的颜色（模拟 Python 的 mostcolor）
;;    作为背景色 (if without-bg? ...)
;; ----------------------------------------------------------------------
(define (argmax lst val-fn)
  (foldl
   (λ (x best)
     (if (> (val-fn x) (val-fn best)) x best))
   (first lst) (rest lst)))

(define (mostcolor grid)
  (define freq (make-hash))
  (for* ([r (in-range (grid-height grid))]
         [c (in-range (grid-width grid))])
    (define col (grid-ref grid (list r c)))
    (hash-update! freq col (λ (old) (add1 old)) 0))
  (define max-pair
    (argmax (hash->list freq)
            (λ (p) (cdr p))))   ; p = (color . count)
  (car max-pair))               ; 返回 color

;; ----------------------------------------------------------------------
;; 3. 定义 4邻接 or 8邻接
;; ----------------------------------------------------------------------
(define (neighbors loc h w)
  "8邻接"
  (define (in-bounds? x y)
    (and (>= x 0) (< x h) (>= y 0) (< y w)))
  (define r (first loc))
  (define c (second loc))
  (filter (λ (xy) (in-bounds? (first xy) (second xy)))
          (list (list r     (add1 c))     ; right
                (list r     (sub1 c))     ; left
                (list (add1 r) c)         ; down
                (list (sub1 r) c)         ; up
                (list (add1 r) (add1 c))  ; diag right-down
                (list (add1 r) (sub1 c))  ; diag left-down
                (list (sub1 r) (add1 c))  ; diag right-up
                (list (sub1 r) (sub1 c)))))

(define (dneighbors loc h w)
  "4邻接"
  (define (in-bounds? x y)
    (and (>= x 0) (< x h) (>= y 0) (< y w)))
  (define r (first loc))
  (define c (second loc))
  (filter (λ (xy) (in-bounds? (first xy) (second xy)))
          (list (list r (add1 c))    ; right
                (list r (sub1 c))    ; left
                (list (add1 r) c)    ; down
                (list (sub1 r) c)))) ; up


;; 假定已有以下辅助函数/结构:
;; (struct Grid (rows) #:transparent)
;; (define (grid-height g) ...)
;; (define (grid-width g) ...)
;; (define (grid-ref g loc) ...)
;; (define (mostcolor g) ...) ; 返回出现最多的颜色
;; (define (neighbors loc h w) ...) ; 8邻接
;; (define (dneighbors loc h w) ...) ; 4邻接

;; 存储与对象相关的各种信息

(provide          (struct-out ObjectInfo) )



(struct ObjectInfo
  (obj               ;; set of '(color (row col))'
   univalued?        ;; 是否仅保留单一颜色
   diagonal?         ;; 是否允许对角扩散
   without-bg?       ;; 是否需要排除背景色
   origin-color      ;; 如果对象保持单色则为种子颜色，否则 -1 或 'multi'
   origin-position   ;; 对象的最小(row,col)，或别的定位
   otherinfo)
  #:transparent)

(provide objects)

(define (objects grid univalued? diagonal? without-bg?)
  ;; 1) 根据 without-bg? 确定背景色
  (define bg
    (if without-bg?
        (mostcolor grid)  ;; 你已有的获取“最常见颜色”的函数
        #f))

  (define h (grid-height grid))
  (define w (grid-width grid))

  ;; 把所有坐标 [(0 0), (0 1) ... (h-1 w-1)] 放入一个 list
  (define locs
    (for*/list ([i (in-range h)]
                [j (in-range w)])
      (list i j)))

  ;; 已占用的坐标
  (define occupied (set))

  ;; 最终返回的对象集 (每个元素是一个 ObjectInfo)
  (define objs (set))

  ;; 若 diagonal?=#t，允许对角；否则仅四邻居
  (define neigh-fn
    (if diagonal? neighbors dneighbors))

  ;; 遍历每个坐标
  (for ([loc (in-list locs)])
    (unless (set-member? occupied loc)
      (define seed-color (grid-ref grid loc))
      ;; 若 seed-color == bg 并且 without-bg?=#t，则跳过
      (unless (and bg (equal? seed-color bg))
        ;; 准备 BFS/DFS
        (define obj (set (list seed-color loc))) ;; 该对象起步
        (define cands (set loc))                  ;; 待扩展坐标
        ;; BFS 种子颜色
        (define origin-color seed-color)
        ;; 标志：是否出现多颜色
        (define multi-color? #f)
        ;; BFS
        (let loop ([c cands]
                   [o obj]
                   [found-colors (set seed-color)])
          (cond
            [(set-empty? c)
             ;; BFS结束 => 求 min-row, min-col
             (define o-list (set->list o)) ; e.g. '((color (r c)) ...)
             (define minr (apply min (map (lambda (x) (first (cadr x))) o-list)))
             (define minc (apply min (map (lambda (x) (second (cadr x))) o-list)))

             ;; 若 multi-color?=#t，则最终 origin-color = -1
             (define final-color
               (if multi-color? -1 origin-color))

             ;; 加入 objs 集
             (set! objs
                   (set-add objs
                            (ObjectInfo
                             o
                             univalued?
                             diagonal?
                             without-bg?
                             final-color
                             (list minr minc)
                             #f)))  ; otherinfo => #f 占位
             ]

            [else
             (define neighborhood (set))
             ;; 遍历 c 里的每个坐标
             (for ([cand (in-set c)])
               (define ccolor (grid-ref grid cand))
               (define add?
                 (if univalued?
                     ;; 单色 => ccolor 必须 == seed-color
                     (equal? ccolor seed-color)
                     ;; 多色 => 只要 != bg
                     (not (and bg (equal? ccolor bg)))))

               (when add?
                 ;; 将坐标加进对象
                 (set! o (set-add o (list ccolor cand)))
                 ;; 标记占用
                 (set! occupied (set-add occupied cand))

                 ;; 如果尚未 multi-color?，且发现了新的颜色 => multi-color? = #t
                 (when (and (not multi-color?)
                            (not (set-member? found-colors ccolor)))
                   (set! multi-color? #t))

                 (set! found-colors (set-add found-colors ccolor))

                 ;; 扩展邻居
                 (define neighs (neigh-fn cand h w))
                 (for ([n neighs])
                   (set! neighborhood (set-add neighborhood n)))))

             ;; 继续下一个 BFS 轮次
             (loop (set-subtract neighborhood occupied)
                   o
                   found-colors)])))))
  ;;; (displayln objs)
  objs)
