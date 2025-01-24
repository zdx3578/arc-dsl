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

(provide objects)

(define (objects grid univalued? diagonal? without-bg?)
  ;; 1) 确定背景色；若 without-bg?=#t，则最常见颜色；否则 #f
  ;;; (displayln (format " > > objects fun log  param = ~a, =~a, =~a"
  ;;;                            univalued? diagonal? without-bg?))
  (define bg
    (if without-bg?
        (mostcolor grid)
        #f)
  ) ;; ← 对应 (define bg

  (define h (grid-height grid))
  (define w (grid-width grid))

  ;; 所有坐标
  (define locs
    (for*/list ([i (in-range h)]
                [j (in-range w)])
      (list i j))
  ) ;; ← 对应 (define locs

  ;; 记录已经属于某个对象的坐标
  (define occupied (set))

  ;; 存放所有对象
  (define objs (set))

  ;; 根据 diagonal? 确定邻居函数
  (define neigh-fn
    (if diagonal?
        neighbors
        dneighbors)
  ) ;; ← 对应 (define neigh-fn

  ;; 主循环，遍历所有坐标
  (for ([loc (in-list locs)])
    (when (not (set-member? occupied loc))
      (define val (grid-ref grid loc))
      ;; 如果是背景色，且要排除，则跳过
      (when (not (and bg (equal? val bg)))
        ;; 创建一个新对象，包含当前格子 (color, (r c))
        (define obj (set (list val loc)))
        ;; 待扩展坐标
        (define cands (set loc))

        ;; 用 let + 递归 loop，模拟 Python while
        (let loop ([c cands]
                   [o obj])
          (cond
            ;; 若候选坐标为空 => BFS/DFS 完成
            [(set-empty? c)
             ;; 将当前对象加入 objs
             (set! objs (set-add objs o))
            ] ;; ← 对应 [(set-empty? c)

            [else
             ;; neighborhood: 本轮找到的周边坐标
             (define neighborhood (set))

             ;; 遍历当前候选
             (for ([cand (in-set c)])
               (define ccolor (grid-ref grid cand))

               ;; 判断是否加入此对象
               (define add?
                 (if univalued?
                     (equal? ccolor val)
                     ;; 若不需单色，则只要不是 bg (when without-bg?=#t)
                     (not (and bg (equal? ccolor bg))))
               ) ;; ← 对应 (define add?

               (when add?
                 ;; 把 (ccolor, cand) 加入对象
                 (set! o (set-add o (list ccolor cand)))
                 ;; 标记占用
                 (set! occupied (set-add occupied cand))
                 ;; 找邻居并放入 neighborhood
                 (define neighs (neigh-fn cand h w))
                 (for ([n neighs])
                   (set! neighborhood (set-add neighborhood n))
                 ) ;; ← 对应 (for ([n neighs])
               ) ;; ← 对应 (when add?
             ) ;; ← 对应 (for ([cand (in-set c)])

             ;; 新一轮候选 = neighborhood - occupied
             (loop (set-subtract neighborhood occupied) o)
            ] ;; ← 对应 [else
          ) ;; ← 对应 (cond
        ) ;; ← 对应 (let loop
      ) ;; ← 对应 (when (not (and bg...
    ) ;; ← 对应 (when (not (set-member...
  ) ;; ← 对应 (for ([loc (in-list locs)])
  ;; 返回所有对象 (每个对象也是 set)
  ;;; (displayln "-----------objects function-----------")
  ;;; (displayln objs)
  objs
) ;; ← 对应 (define (objects ...