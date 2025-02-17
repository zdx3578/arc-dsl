#lang rosette

(require racket/set
         racket/list
        ;;;  "data-structures.rkt"
         "objects.rkt")

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









(define (removef item lst)
  (filter (lambda (x) (not (eq? x item))) lst))





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

; ;; (hmirror obj) => 上下翻转 (horizontal mirror)
; ;; row' = (rmax + rmin) - row, col' = col
(define (hmirror obj)
  (unless (object? obj)
    (error "hmirror: expected object but got" obj))
  (define-values (rmin rmax cmin cmax) (object-bbox obj))
  (for/set ([e (in-set obj)])
    (define cval (first e))
    (define ro (first (second e)))
    (define co (second (second e)))
    (list cval (list (- (+ rmax rmin) ro) co))))

; (define (hmirror obj grid-height)
;   (unless (object? obj)
;     (error "hmirror: expected object but got" obj))
;   ;; 假设行坐标从 0 到 (grid-height - 1)
;   (for/set ([e (in-set obj)])
;     (define cval (first e))
;     (define ro (first (second e)))
;     (define co (second (second e)))
;     (list cval (list (- (sub1 grid-height) ro) co))))


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


;; -------------------------------------------------------
;; 1) 定义一些辅助：object 宽/高，以及各个操作的有效性检查
;; -------------------------------------------------------

;; object-width / object-height 仅作演示，
;; 通过 object-bbox 来拿到 min/max 行列，再计算宽高。
(define (object-width obj)
  (define-values (rmin rmax cmin cmax) (object-bbox obj))
  (if (set-empty? obj)
      0
      (add1 (- rmax rmin)))) ;; 行的范围(含端点) => 宽

(define (object-height obj)
  (define-values (rmin rmax cmin cmax) (object-bbox obj))
  (if (set-empty? obj)
      0
      (add1 (- cmax cmin)))) ;; 列的范围(含端点) => 高

; ;; 判断 Rot90 的可行性(这里只是示例, 你可换成别的规则)
; (define (valid-rot90? obj)
;   (and (object? obj)
;        (not (set-empty? obj))          ;; 不要空对象
;        (> (object-width obj) 1)
;        (> (object-height obj) 1)))

; ;; 判断 HMirror 的可行性
; (define (valid-hmirror? obj)
;   (and (object? obj)
;        (not (set-empty? obj))
;        (> (object-width obj) 0)
;        ;; 当然你可以添加更多逻辑……
;        #t))

; ;; 判断 VMirror 的可行性
; (define (valid-vmirror? obj)
;   (and (object? obj)
;        (not (set-empty? obj))
;        (> (object-height obj) 0)
;        #t))



; ;; 对象判定 (供示例使用)
; (define (object? s)
;   (and (set? s)
;        (for/and ([e (in-set s)])
;          (and (list? e)
;               (= (length e) 2)
;               (list? (second e))
;               (= (length (second e)) 2)))))
; ;; 简单校验: 每个元素形如 (colorValue (r c))

;;=====================================================
;; 1) vmirror-set: 垂直镜像(左右翻转)
;;    (r, c) => (r, (grid-width - 1 - c))
;;=====================================================
(define (vmirror-set obj grid-width)
  (unless (object? obj)
    (error "vmirror-set: expected an object set but got" obj))
  (for/set ([e (in-set obj)])
    (define color (first e))
    (define ro    (first (second e)))
    (define co    (second (second e)))
    (list color (list ro (- (sub1 grid-width) co)))))


;;=====================================================
;; 2) hmirror-set: 水平镜像(上下翻转)
;;    (r, c) => ((grid-height - 1 - r), c)
;;=====================================================
(define (hmirror-set obj grid-height)
  (unless (object? obj)
    (error "hmirror-set: expected an object set but got" obj))
  (for/set ([e (in-set obj)])
    (define color (first e))
    (define ro    (first (second e)))
    (define co    (second (second e)))
    (list color (list (- (sub1 grid-height) ro) co))))


;;=====================================================
;; 3) dmirror-set: 沿主对角线 (左上 -> 右下) 翻转
;;    通常 (r, c) => (c, r)
;;    若 grid 高宽不一样，此操作可能导致出界，需自行处理
;;=====================================================
(define (dmirror-set obj grid-height grid-width)
  (unless (object? obj)
    (error "dmirror-set: expected an object set but got" obj))
  (for/set ([e (in-set obj)])
    (define color (first e))
    (define ro    (first (second e)))
    (define co    (second (second e)))
    ;; 简单做 (r, c) => (c, r)
    ;; 若 grid 高度 != 宽度，可能产生无效坐标(超出grid边界)
    (list color (list co ro))))


;;=====================================================
;; 4) cmirror-set: 沿反对角线 (右上->左下) 翻转
;;    若我们设想反对角线是 ↗，通常公式:
;;    (r, c) => ( (grid-height - 1 - c),
;;                (grid-width  - 1 - r) )
;;=====================================================
(define (cmirror-set obj grid-height grid-width)
  (unless (object? obj)
    (error "cmirror-set: expected an object set but got" obj))
  (for/set ([e (in-set obj)])
    (define color (first e))
    (define ro    (first (second e)))
    (define co    (second (second e)))
    (list color
          (list (- (sub1 grid-height) co)
                (- (sub1 grid-width)  ro)))))


;;=====================================================
;; 5) rotate90-set: 逆时针 or 顺时针 90?
;;    这里示例: (r, c) => (c, (grid-height - 1 - r))
;;    表示逆时针旋转90(以左上角为原点)
;;=====================================================
(define (rotate90-set obj grid-height grid-width)
  (unless (object? obj)
    (error "rotate90-set: expected an object set but got" obj))
  (for/set ([e (in-set obj)])
    (define color (first e))
    (define ro    (first (second e)))
    (define co    (second (second e)))
    ;; 旋转公式: newR = co, newC = (grid-height - 1 - ro)
    (list color (list co (- (sub1 grid-height) ro)))))


;;=====================================================
;; 6) rotate180-set: 180 度 = 两次 90 或直接公式
;;    这里直接用 (r, c) => (grid-height-1 - r, grid-width-1 - c)
;;=====================================================
(define (rotate180-set obj grid-height grid-width)
  (unless (object? obj)
    (error "rotate180-set: expected an object set but got" obj))
  (for/set ([e (in-set obj)])
    (define color (first e))
    (define ro    (first (second e)))
    (define co    (second (second e)))
    (list color
          (list (- (sub1 grid-height) ro)
                (- (sub1 grid-width)  co)))))



(define id-manager%
  (class object%
    ;; 初始化字段
    (super-new)
    (define tables (make-hash)) ; e.g. tables['shape'] = {'shape_1': 1, 'shape_2': 2, ...}
    (define next-id (make-hash)) ; e.g. next_id['shape'] = 1

    ;; 获取 ID 的方法
    (define/public (get-id category value)
      ;; 如果 category 不存在，则初始化
      (when (not (hash-has-key? tables category))
        (hash-set! tables category (make-hash))
        (hash-set! next-id category 1))

      ;; 获取当前 category 的表
      (let ([category-table (hash-ref tables category)])
        ;; 如果 value 不存在，则分配新的 ID
        (when (not (hash-has-key? category-table value))
          (hash-set! category-table value (hash-ref next-id category))
          (hash-set! next-id category (+ (hash-ref next-id category) 1)))

        ;; 返回对应的 ID
        (hash-ref category-table value)))

    ;; 打印所有对象及其 ID 的方法
    (define/public (print-all-ids)
      (for ([(category category-table) (in-hash tables)]) ; 遍历每个类别
        (displayln (format "\n\nCategory: ~a, length: ~a \n" category (hash-count category-table))) ; 打印类别名称
        (for ([(value id) (in-hash category-table)]) ; 遍历类别中的对象及其 ID
          (displayln (format "ID : ~a -> Object content -> : \n                  ~a" id value )))))

    (define/public (reset)
      ;; 清空 tables 和 next-id
      (set! tables (make-hash))
      (set! next-id (make-hash))
      (displayln "All data has been reset."))))



(define (ObjInf->obj+grid objinfo)
  (define oldobj (ObjInf-obj objinfo))
  (define grid-size (ObjInf-grid-H-W objinfo))
  (define gheight (first grid-size))
  (define gwidth  (second grid-size))
  (values oldobj gheight gwidth))

(define managerid (new id-manager%))

(define (makeshift-ObjInf objinfo obj00 obj000 )
  (ObjInf
    (ObjInf-pair-id objinfo)
    (ObjInf-in-or-out objinfo)
    (ObjInf-configparam objinfo)
    (ObjInf-obj objinfo)
    obj00
    (send managerid get-id "OBJshape"  obj000 )
    obj000
    (ObjInf-grid-H-W objinfo)
    (ObjInf-bounding-box objinfo)
    (ObjInf-color-ranking objinfo)
    (ObjInf-otherinfo objinfo)))

(define (update-objinfo-obj objinfo newobj)
  (ObjInf
    (ObjInf-pair-id objinfo)
    (ObjInf-in-or-out objinfo)
    (ObjInf-configparam objinfo)
    newobj
    (ObjInf-obj-00 objinfo)
    (ObjInf-obj-ID objinfo)
    (ObjInf-obj-000 objinfo)
    (ObjInf-grid-H-W objinfo)
    (ObjInf-bounding-box objinfo)
    (ObjInf-color-ranking objinfo)
    (ObjInf-otherinfo objinfo)))

(define (smallnoobj-objinfo-obj objinfo )
  (ObjInf
    (ObjInf-pair-id objinfo)
    (ObjInf-in-or-out objinfo)
    (ObjInf-configparam objinfo)
    ""
    ""
    (ObjInf-obj-ID objinfo)
    ""
    (ObjInf-grid-H-W objinfo)
    (ObjInf-bounding-box objinfo)
    (ObjInf-color-ranking objinfo)
    (ObjInf-otherinfo objinfo)))


(define (hmirror-info objinfo)
  (unless (ObjInf? objinfo)
    (error "hmirror-info: expected ObjInf but got" objinfo))
  (define-values (oldobj gh gw) (ObjInf->obj+grid objinfo))
  (define newobj (hmirror-set oldobj gh))
  (update-objinfo-obj objinfo newobj))

;;=====================================================
;; 1) vmirror-info
(define (vmirror-info objinfo)
  (unless (ObjInf? objinfo)
    (error "vmirror-info: expected ObjInf but got" objinfo))
  (define-values (oldobj gh gw) (ObjInf->obj+grid objinfo))
  (define newobj (vmirror-set oldobj gw))
  (update-objinfo-obj objinfo newobj))

;;=====================================================
;; 3) dmirror-info
;;=====================================================
(define (dmirror-info objinfo)
  (unless (ObjInf? objinfo)
    (error "dmirror-info: expected ObjInf but got" objinfo))
  (define-values (oldobj gh gw) (ObjInf->obj+grid objinfo))
  (define newobj (dmirror-set oldobj gh gw))
  (update-objinfo-obj objinfo newobj))

;;=====================================================
;; 4) cmirror-info
;;=====================================================
(define (cmirror-info objinfo)
  (unless (ObjInf? objinfo)
    (error "cmirror-info: expected ObjInf but got" objinfo))
  (define-values (oldobj gh gw) (ObjInf->obj+grid objinfo))

  (define newobj (cmirror-set oldobj gh gw))
  (update-objinfo-obj objinfo newobj))

;;=====================================================
;; 5) rotate90-info
;;=====================================================
(define (rotate90-info objinfo)
  (unless (ObjInf? objinfo)
    (error "rotate90-info: expected ObjInf but got" objinfo))
  (define-values (oldobj gh gw) (ObjInf->obj+grid objinfo))

  (define newobj (rotate90-set oldobj gh gw))
  (update-objinfo-obj objinfo newobj))

;;=====================================================
;; 6) rotate180-info
;;=====================================================
(define (rotate180-info objinfo)
  (unless (ObjInf? objinfo)
    (error "rotate180-info: expected ObjInf but got" objinfo))
  (define-values (oldobj gh gw) (ObjInf->obj+grid objinfo))

  (define newobj (rotate180-set oldobj gh gw))
  (update-objinfo-obj objinfo newobj))




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



;; -------------------------------------------------------
;; 5) 8 种布尔参数组合 & 汇总生成
;; -------------------------------------------------------
(define param-combinations
  (list
   (list #f #f #f)
   (list #f #f #t)
   (list #f #t #f)
   (list #f #t #t)
   (list #t #f #f)
   (list #t #f #t)
   (list #t #t #f)
   (list #t #t #t)))


  ;;;  (define param-combinations
  ;;; (list
  ;;;  (list #t #t #t)
  ;;;  (list #f #f #t)
  ;;;  (list #t #f #t)
  ;;;  (list #f #t #t)
  ;;;  (list #t #f #f)
  ;;;  (list #f #t #f)
  ;;;  (list #t #t #f)
  ;;;  (list #f #f #f )))

(define (objects-with-params the-pair-id in-or-out grid bools)
  (define b1 (list-ref bools 0))
  (define b2 (list-ref bools 1))
  (define b3 (list-ref bools 2))
  (objects the-pair-id in-or-out grid b1 b2 b3)) ;; 根据你的实际签名调整

(define (all-objects-from-grid the-pair-id in-or-out grid)
  (for/fold ([acc (set)])
            ([params (in-list param-combinations)])
    (set-union acc (objects-with-params the-pair-id in-or-out grid params))))

; (define (all-objects-from-grid-withra-ids grid pair-id)
;   (define counter 0)
;   (define (annotate-object obj)
;     (let* ([unique-id (format "~a-obj-~a" pair-id counter)]
;            [new-obj (hash-set obj 'unique-id unique-id)])
;       (set! counter (add1 counter))
;       new-obj))
;   (for/fold ([acc (set)])
;             ([params (in-list param-combinations)])
;     (set-union acc (set-map annotate-object (objects-with-params grid params)))))


;; 4. 将一组坐标平移到 (0, 0) 的辅助函数。
;;    例如，如果最小行号是 minr，最小列号是 minc，
;;    那么将所有 (r, c) 转换为 (r - minr, c - minc)。
;; -------------------------------------------------------------------
(define (shift-coords-to-0-0 coords)
  (unless (set-empty? coords)
    ;; 如果不空，就执行正常逻辑
    (define coords-list (set->list coords))
    (define min-row (apply min (map first coords-list)))
    (define min-col (apply min (map second coords-list)))
    (set
     (for/list ([rc (in-list coords-list)])
       (let ([r (first rc)]
             [c (second rc)])
         (cons (- r min-row) (- c min-col)))))))
;; -------------------------------------------------------------------
;; 5. 组合所有步骤，得到 “对象形状” 的集合。
;;    每个对象形状是一个“平移后”的坐标集合 (以 0,0 为左上角)。
;; -------------------------------------------------------------------
;;; (define (all-objects-shape-from-grid grid)
;;;   (define all-objs (all-objects-from-grid grid))
;;;   (for/set ([obj (in-set all-objs)])
;;;     (define coords (asindices obj))
;;;     (shift-coords-to-0-0 coords)))
(define (all-00shape-from-objs all-objs)
  ;;; (define all-objs (all-objects-from-grid grid))
  (for/set ([obj (in-set all-objs)])
    (define coords (asindices obj))
    (shift-coords-to-0-0 coords)))

(define (all-shape-from-objs all-objs)
  ;;; (define all-objs (all-objects-from-grid grid))
  (for/set ([obj (in-set all-objs)])
    (asindices obj)))
    ;;; (define coords (asindices obj))
    ;;; (shift-coords-to-0-0 coords)))



;; 假设结构体定义类似:
;; (struct ObjInf
;;   (obj univalued? diagonal? without-bg? origin-color origin-position otherinfo)
;;   #:transparent)



(define (shift-obj-to-0-0-0 objbig)
  ;; Step 1: 判断 objbig 是否 ObjInf
  (cond
    [(ObjInf? objbig)
      ;; 从 objbig 中提取原 BFS 集合
      (define orig-obj (ObjInf-obj objbig))
      (define obj000 (shift-pure-obj-to-0-0-0 orig-obj))
      (define obj00 (shift-pure-obj-to-00 orig-obj))

      (makeshift-ObjInf objbig obj00 obj000 )]

    [(set? objbig)
     ;; 如果传入的是纯 set，直接平移
      (shift-pure-obj-to-0-0-0 objbig)]
    [else
     (error "shift-obj-to-0-0-0: unsupported argument type" objbig)]))

;; move to obj funciton  把“纯对象集合(set)”平移到 (0,0)，并把颜色设为 0
(define (shift-pure-obj-to-0-0-0 obj)
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
          (list 0 (list (- r min-row) (- c min-col)))))))

(define (shift-pure-obj-to-00 obj)
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


; (define sameshapediffcolor)

; (define sameobj-absolute)

; (define sameobj-sameshape-diffcolor-diffpostion)

; (define sameobj-sameshape-samecolor?-samepostion?)

; (define sameobj-samepostion-diffcolor)



(define (all-objects-00-c0-from-objs all-objs)
  ;;; (define all-objs (all-objects-from-grid grid))
  (for/set ([obj (in-set all-objs)])
    ;;; (define coords (asindices obj))
    (shift-obj-to-0-0-0 obj)))







(define (display-param-records records)
  ;; 外层循环：遍历 local-param-records 的每个元素
  (for-each
    (lambda (record)
      ;; 打印当前记录（调试用）
      (displayln (format "\n  ~s" record))

      ;; 判断当前记录是否包含子元素
      (when (list? record) ; 假设子元素是列表
        ;; 内层循环：遍历子元素
        (for-each
          (lambda (sub-record)
            (displayln (format "\n  sub-record: ~s" sub-record))
            ;; 在这里可以对子元素进行进一步处理
            )
          record)))
    records))


;; 示例用法
; (define manager (new id-manager%)

; ;; 单个值的情况
; (send manager get-id "shape" "shape_1") ; 返回 1
; (send manager get-id "shape" "shape_2") ; 返回 2

; ;; 集合的情况
; (define shape-set-1 (set "shape_3" "shape_4"))
; (define shape-set-2 (set "shape_5" "shape_6"))

; (send manager get-id "shape" shape-set-1) ; 返回 3
; (send manager get-id "shape" shape-set-2) ; 返回 4

; ;; 打印所有对象及其 ID
; (send manager print-all-ids)









;; -------------------------------------------------------
;; 判断较小网格是否是较大网格的一个部分
;;; ;; -------------------------------------------------------
;;; (define (is-subgrid? big-grid small-grid)
;;;   (define big-objects (objects big-grid))
;;;   (define small-objects (objects small-grid))

;;;   ;; 获取大矩阵和小矩阵的尺寸
;;;   (define big-rows (length big-objects))
;;;   (define big-cols (length (first big-objects)))
;;;   (define small-rows (length small-objects))
;;;   (define small-cols (length (first small-objects)))

;;;   ;; 遍历大矩阵，检查是否存在小矩阵匹配的位置
;;;   (for ([i (in-range (- big-rows small-rows + 1))])
;;;     (for ([j (in-range (- big-cols small-cols + 1))])
;;;       (define match #t)
;;;       ;; 检查大矩阵的当前位置是否与小矩阵完全匹配
;;;       (for ([x (in-range small-rows)])
;;;         (for ([y (in-range small-cols)])
;;;           (unless (= (list-ref (list-ref big-objects (+ i x)) (+ j y))
;;;                      (list-ref (list-ref small-objects x) y))
;;;             (set! match #f)))
;;;         (when (not match) (break)))
;;;       (when match
;;;         (return #t)))
;;;   #f)

;;; ;; -------------------------------------------------------
;;; ;; 判断多个小网格是否能拼装成一个大网格
;;; ;; -------------------------------------------------------
;;; (define (can-assemble-grid subgrids big-grid)
;;;   (define big-objects (objects big-grid))
;;;   (define all-small-objects (apply append (map objects subgrids)))

;;;   ;; 遍历小网格，检查是否可以填充大网格
;;;   (define assembled #t)
;;;   (for ([i (in-range (length all-small-objects))])
;;;     (define small-object (list-ref all-small-objects i))
;;;     ;; 在大网格中查找匹配的位置
;;;     (let loop ([big-objects big-objects] [remaining small-object] [found #f])
;;;       (for ([row big-objects])
;;;         (if (not found)
;;;             (for ([col row])
;;;               ;; 判断能否把该小网格拼装到大网格中的位置
;;;               (if (= col remaining)
;;;                   (set! found #t)))))
;;;     )
;;;     (unless found (set! assembled #f)))
;;;   assembled)


;;; ;; 测试
;;; (define big-grid [[16 15 14] [13 17 14] [15 15 16]])
;;; (define small-grid [[16 15] [14 15]])

;;; (displayln (is-subgrid? big-grid small-grid)) ; 应该返回 #t
;;; (displayln (can-assemble-grid (list small-grid small-grid) big-grid)) ; 应该返回 #t

