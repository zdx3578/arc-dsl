#lang rosette

(require racket/set
         racket/list)

(provide (all-defined-out))

;; -----------------------------------------------------------------------------
;; 在本示例中，“对象 (Object)” 统一表示为：
;;    一个 (Setof (List Integer (List Integer Integer)))
;; 也就是：每个元素是形如 '(color (row col)) 的列表，
;; 其中 color、row、col 都是整数。
;;
;; 例如，对象 #<set: (3 (12 7)) (3 (12 6)) ... >
;; 表示颜色为3, 坐标(12,7)、(12,6) 等等。
;; -----------------------------------------------------------------------------

;; -----------------------------------------------------------------------------
;; 1. 帮助判断是否是“对象”（而不是别的值）
;; -----------------------------------------------------------------------------
(define (object? obj)
  (and (set? obj)
       (for/and ([elem (in-set obj)])
         (and (list? elem)
              (= (length elem) 2)
              (integer? (first elem))            ; color
              (list? (second elem))
              (integer? (first (second elem)))   ; row
              (integer? (second (second elem)))))))

;; -----------------------------------------------------------------------------
;; 2. 基础属性：size / shape / asindices / colorcount / palette / numcolors
;; -----------------------------------------------------------------------------

;; (size obj) => 返回对象内元素数量
(define (size obj)
  (unless (object? obj)
    (error "size: expected object but got" obj))
  (set-count obj))

;; 取对象 bounding box 的 [rmin rmax cmin cmax]
(define (object-bbox obj)
  (define rows (for/list ([e (in-set obj)]) (first (second e))))
  (define cols (for/list ([e (in-set obj)]) (second (second e))))
  (values (apply min rows) (apply max rows)
          (apply min cols) (apply max cols)))

;; (shape obj) => (height width)，对象所占 bounding-box 的尺寸
(define (shape obj)
  (unless (object? obj)
    (error "shape: expected object but got" obj))
  (define-values (rmin rmax cmin cmax) (object-bbox obj))
  (list (add1 (- rmax rmin))  ; height
        (add1 (- cmax cmin)))) ; width

;; (asindices obj) => 返回对象所有 (row col) 的坐标集 (不含 color)
(define (asindices obj)
  (unless (object? obj)
    (error "asindices: expected object but got" obj))
  (for/set ([e (in-set obj)])
    (second e)))  ; e = '(color (r c))

;; (colorcount obj colorval) => 统计对象里颜色 == colorval 的格子数
(define (colorcount obj colorval)
  (unless (object? obj)
    (error "colorcount: expected object but got" obj))
  (count (λ (elem) (eq? (first elem) colorval))
         (set->list obj)))

;; (palette obj) => 返回对象内出现的所有颜色 (set-of color)
(define (palette obj)
  (unless (object? obj)
    (error "palette: expected object but got" obj))
  (for/set ([e (in-set obj)])
    (first e)))

;; (numcolors obj) => palette 的大小
(define (numcolors obj)
  (set-count (palette obj)))

;; 若需要排除颜色 0，则比如：
(define (numcolors-nozero obj)
  (define p (palette obj))
  (set-count (set-remove p 0)))


;; -----------------------------------------------------------------------------
;; 3. mostcolor / leastcolor
;; -----------------------------------------------------------------------------

;; 通用 argmax, argmin
(define (argmax lst val-fn)
  (foldl (λ (x best)
           (if (> (val-fn x) (val-fn best)) x best))
         (first lst) (rest lst)))

(define (argmin lst val-fn)
  (foldl (λ (x best)
           (if (< (val-fn x) (val-fn best)) x best))
         (first lst) (rest lst)))

;; (mostcolor obj) => 出现次数最多的颜色
(define (mostcolor obj)
  (unless (object? obj)
    (error "mostcolor: expected object but got" obj))
  (define cols (palette obj))
  (if (set-empty? cols)
      (error "mostcolor: object has no colors" obj)
      (argmax (set->list cols)
              (λ (c) (colorcount obj c)))))

;; (leastcolor obj) => 出现次数最少的颜色
(define (leastcolor obj)
  (unless (object? obj)
    (error "leastcolor: expected object but got" obj))
  (define cols (palette obj))
  (if (set-empty? cols)
      (error "leastcolor: object has no colors" obj)
      (argmin (set->list cols)
              (λ (c) (colorcount obj c)))))

;; -----------------------------------------------------------------------------
;; 4. 各种镜像 (hmirror, vmirror, dmirror, cmirror)
;;    均对 bounding box 做变换: (row,col) -> (row',col')
;; -----------------------------------------------------------------------------

;; (hmirror obj) => 上下翻转 (horizontal mirror)
;; row' = (rmax + rmin) - row, col' = col
(define (hmirror obj)
  (unless (object? obj)
    (error "hmirror: expected object but got" obj))
  (define-values (rmin rmax cmin cmax) (object-bbox obj))
  (for/set ([e (in-set obj)])
    (define cval (first e))
    (define ro (first (second e)))
    (define co (second (second e)))
    (list cval (list (- (+ rmax rmin) ro) co))))

;; (vmirror obj) => 左右翻转 (vertical mirror)
;; row' = row, col' = (cmax + cmin) - col
(define (vmirror obj)
  (unless (object? obj)
    (error "vmirror: expected object but got" obj))
  (define-values (rmin rmax cmin cmax) (object-bbox obj))
  (for/set ([e (in-set obj)])
    (define cval (first e))
    (define ro (first (second e)))
    (define co (second (second e)))
    (list cval (list ro (- (+ cmax cmin) co)))))

;; (dmirror obj) => 沿正对角线翻转 (对 bounding box 的左上->右下)
;; (r, c) -> (rmin + (c-cmin), cmin + (r-rmin))
(define (dmirror obj)
  (unless (object? obj)
    (error "dmirror: expected object but got" obj))
  (define-values (rmin rmax cmin cmax) (object-bbox obj))
  (for/set ([e (in-set obj)])
    (define cval (first e))
    (define ro (first (second e)))
    (define co (second (second e)))
    (list cval
          (list (+ rmin (- co cmin))
                (+ cmin (- ro rmin))))))

;; (cmirror obj) => 沿反对角线翻转 (左下->右上)
;; 一种简易方式: = vmirror(dmirror(vmirror obj)) 或自己写公式
(define (cmirror obj)
  (vmirror (dmirror (vmirror obj))))

;; -----------------------------------------------------------------------------
;; 5. 旋转 90/180 (如需 270可再叠加)
;; -----------------------------------------------------------------------------

;; rotate90: bounding box-based 90度旋转
;; local (r, c) => (c, height-1-r)，然后映射回全局
(define (rotate90 obj)
  (unless (object? obj)
    (error "rotate90: expected object but got" obj))
  (define-values (rmin rmax cmin cmax) (object-bbox obj))
  (define nr (add1 (- rmax rmin))) ;; bounding-box 高度
  ;; (nc (add1 (- cmax cmin))) ;; 宽度 - 如果需要可用
  (for/set ([e (in-set obj)])
    (define colr (first e))
    (define oldr (first (second e)))
    (define oldc (second (second e)))
    ;; 移动到 (rmin,cmin) 为基准
    (define local-r (- oldr rmin))
    (define local-c (- oldc cmin))
    ;; 旋转公式 => newr = local-c, newc = (nr - 1 - local-r)
    (define newr local-c)
    (define newc (- (sub1 nr) local-r))
    ;; 再加回 (rmin, cmin)
    (list colr (list (+ rmin newr) (+ cmin newc)))))

;; rotate180: 两次90 或直接坐标公式
(define (rotate180 obj)
  (rotate90 (rotate90 obj)))
  ;; 或者用公式也行:
  ;; (define-values (rmin rmax cmin cmax) (object-bbox obj))
  ;; ...

;; -----------------------------------------------------------------------------
;; 使用示例（如要测试）:
;; (define test-obj
;;   (set '( (3 (0 0))
;;           (3 (0 1))
;;           (3 (1 0))
;;           (3 (1 1)) )))
;;
;; (displayln (size test-obj))              ; => 4
;; (displayln (shape test-obj))             ; => '(2 2)
;; (displayln (asindices test-obj))         ; => #<set: (0 0) (0 1) (1 0) (1 1)>
;; (displayln (palette test-obj))           ; => #<set: 3>
;; (displayln (rotate90 test-obj))
;; (displayln (hmirror test-obj))
;; (displayln (cmirror test-obj))
;; ...

(define (toindices piece)
  (cond
    ;;; [(grid? piece)
    ;;;  (asindices piece)]
    [(object? piece)
     ;; 只取坐标
     (for/set ([e (in-set piece)])
       (second e))]
    [else
     (error "toindices: not a grid or object" piece)]))

(define (ulcorner piece)
  (define idx (toindices piece))
  (list (apply min (for/list ([p (in-set idx)]) (first p)))
        (apply min (for/list ([p (in-set idx)]) (second p)))))

(define (lrcorner piece)
  (define idx (toindices piece))
  (list (apply max (for/list ([p (in-set idx)]) (first p)))
        (apply max (for/list ([p (in-set idx)]) (second p)))))

;; backdrop: bounding-box indices
(define (backdrop patch)
  (define idx (toindices patch))
  (if (set-empty? idx)
      (set)
      (begin
        (define si (apply min (for/list ([p (in-set idx)]) (first p))))
        (define ei (apply max (for/list ([p (in-set idx)]) (first p))))
        (define sj (apply min (for/list ([p (in-set idx)]) (second p))))
        (define ej (apply max (for/list ([p (in-set idx)]) (second p))))
        (for/set ([i (in-range si (add1 ei))]
                  [j (in-range sj (add1 ej))])
          (list i j)))))
