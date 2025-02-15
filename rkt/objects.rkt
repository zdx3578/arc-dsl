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

(provide          (struct-out ObjInf) )

(provide objects)


;; -----------------------
;; 1) 新的 ObjInf 结构体
;; -----------------------



;; 辅助函数: BFS 结束后, 根据对象 => 求 bounding-box
(define (compute-bbox obj-set)
  ;; obj_set 里每个元素形如 '(color (r c))
  ;; 如果 obj_set 为空, 就返回 '(0 0 0 0) or '(#f #f #f #f)
  (if (set-empty? obj-set)
      '(0 0 0 0)
      (let* ([o-list (set->list obj-set)]
             [rows (map (lambda (x) (first (cadr x))) o-list)]
             [cols (map (lambda (x) (second (cadr x))) o-list)]
             [minr (apply min rows)]
             [maxr (apply max rows)]
             [minc (apply min cols)]
             [maxc (apply max cols)])
        (list minr minc maxr maxc))))

;; 辅助函数: 统计颜色频次 => 返回按降序排列的字符串
(define (color-frequency-ranking obj-set)
  (if (set-empty? obj-set)
      ;; 如果对象为空，返回空列表
      '()
      (let ([color-hash (make-hash)])
        ;; 1) 统计每种颜色出现次数
        (for ([elem (in-set obj-set)])
          ;; elem 形如 '(color (r c))，取第一个就是 color
          (define col (first elem))
          (hash-update! color-hash col add1 0))

        ;; 2) 转为可排序列表：形如 '((color1 . count1) (color2 . count2) ...)
        (define color-count-list
          (hash-map color-hash (lambda (k v)
                                 (cons k v))))

        ;; 3) 按 count 降序排序
        (define sorted
          (sort color-count-list
                (lambda (a b)
                  (> (cdr a) (cdr b)))))

        ;; 4) 将每个元素转成 '(color count) 形式返回
        (map (lambda (pair)
               (list (car pair) (cdr pair)))
             sorted))))

(define (shift-pure-obj-to-0-0 obj)
  (if (set-empty? obj)
      (set)
      (let* ([obj-list (set->list obj)]
             [rc-list  (map (λ(e) (cadr e)) obj-list)]
             [min-row  (apply min (map first rc-list))]
             [min-col  (apply min (map second rc-list))])
        (for/set ([e (in-list obj-list)])
          ;; e = '(color (r c))
          (define color (first e))
          (define r (first (cadr e)))
          (define c (second (cadr e)))
          (list color (list (- r min-row) (- c min-col)))))))



;; 主函数: objects
(define (objects the-pair-id in-or-out grid univalued? diagonal? without-bg?)
  (define bg
    (if without-bg?
        (mostcolor grid)
        #f))
  (define h (grid-height grid))
  (define w (grid-width grid))

  (define locs
    (for*/list ([i (in-range h)]
                [j (in-range w)])
      (list i j)))

  (define occupied (set))
  (define objs (set))

  (define neigh-fn
    (if diagonal? neighbors dneighbors))

  (for ([loc (in-list locs)])
    (unless (set-member? occupied loc)
      (define seed-color (grid-ref grid loc))
      (unless (and bg (equal? seed-color bg))
        (define obj (set (list seed-color loc)))
        (define cands (set loc))
        (define origin-color seed-color)
        (define multi-color? #f)

        (let loop ([c cands]
                   [o obj]
                   [found-colors (set seed-color)])
          (cond
            [(set-empty? c)
             ;; BFS结束
             ;; => 1) bounding box
             (define bounding (compute-bbox o))
             ;; => 2) 若 multi-color?=#t => origin-color = -1
             (define final-color (if multi-color? -1 origin-color))
             ;; => 3) 颜色频次排序
             (define rank (color-frequency-ranking o))

             (define o-list (set->list o))
             (define minr (first bounding))
             (define minc (second bounding))

             (set! objs
                   (set-add objs
                            (ObjInf
                            the-pair-id
                            in-or-out
                            (list  univalued?                              diagonal?                             without-bg?)
                            o
                            #f
                             #f
                            #f
                            (list  h w  )
                            bounding
                            rank
                            #f)))]

            [else
             (define neighborhood (set))
             (for ([cand (in-set c)])
               (define ccolor (grid-ref grid cand))
               (define add?
                 (if univalued?
                     (equal? ccolor seed-color)
                     (not (and bg (equal? ccolor bg)))))
               (when add?
                 (set! o (set-add o (list ccolor cand)))
                 (set! occupied (set-add occupied cand))
                 (when (and (not multi-color?)
                            (not (set-member? found-colors ccolor)))
                   (set! multi-color? #t))
                 (set! found-colors (set-add found-colors ccolor))
                 (define neighs (neigh-fn cand h w))
                 (for ([n neighs])
                   (set! neighborhood (set-add neighborhood n)))))

             (loop (set-subtract neighborhood occupied)
                   o
                   found-colors)])))))
  ; (displayln "from obj fun")
  objs)


(struct ObjInf
  ( pair-id
    in-or-out
    configparam
    obj
    obj-00
    obj-ID
    obj-000
    grid-H-W
    bounding-box        ;; (list minr minc maxr maxc)
    color-ranking       ;; string or list
    otherinfo)
  #:transparent)