#lang rosette

(provide (all-defined-out))

;;; o1 generation
;;; // ...existing code...

;; 将 Patch 或 Object 转换为坐标集合
(define (toindices element)
  (cond
    [(set? element)
     ;; 如果是 Object 则形如 (color, (row col))，或者直接 (row col)
     (define first-item (first (set->list element)))
     (if (and (pair? first-item) (pair? (second first-item)))
         (set-map element (λ (x) (second x)))  ;; (v, (r, c)) => (r, c)
         element)]
    [(tuple? element)
     ;; 如果是 Grid，则将每一个坐标加入集合
     (define rows (tuple->list element))
     (define h (length rows))
     (define w (length (first rows)))
     (for*/set ([r (in-range h)]
                [c (in-range w)])
       (list r c))]
    [else
     (error "Unknown element type")]))

;; 计算对象大小
(define (size obj)
  (set-count obj))

;; 计算对象形状 (height, width)
;; 注意：按 bounding box 计算
(define (shape element)
  (define indices (toindices element))
  (if (set-empty? indices)
      '(0 0)
      (begin
        (define si (apply min (map first (set->list indices))))
        (define ei (apply max (map first (set->list indices))))
        (define sj (apply min (map second (set->list indices))))
        (define ej (apply max (map second (set->list indices))))
        (list (+ 1 (- ei si)) (+ 1 (- ej sj))))))

;; 计算 palette
(define (palette element)
  (if (tuple? element)
      (list->set (flatten (tuple->list element)))
      (list->set (map first (set->list element)))))

;; 计算颜色总数
(define (numcolors element)
  (set-count (palette element)))

;; 计算特定颜色数量
(define (colorcount element color)
  (cond
    [(tuple? element)
     (length (filter (λ (v) (equal? v color)) (flatten (tuple->list element))))]
    [else
     (length (filter (λ (x) (equal? (first x) color)) (set->list element)))]))

;; 最常见颜色
(define (mostcolor element)
  (define vals
    (if (tuple? element)
        (flatten (tuple->list element))
        (map first (set->list element))))
  (define freq (make-hash))
  (for ([v vals])
    (hash-set! freq v (add1 (hash-ref freq v 0))))
  (argmax (λ (k) (hash-ref freq k)) (hash-keys freq)))

;; 最少见颜色
(define (leastcolor element)
  (define vals
    (if (tuple? element)
        (flatten (tuple->list element))
        (map first (set->list element))))
  (define freq (make-hash))
  (for ([v vals])
    (hash-set! freq v (add1 (hash-ref freq v 0))))
  (argmin (λ (k) (hash-ref freq k)) (hash-keys freq)))

;; 镜像示例 (水平镜像)
(define (hmirror piece)
  (if (tuple? piece)
      (reverse piece)
      (let ([d (+ (first (ulcorner piece)) (first (lrcorner piece)))])
        (if (and (pair? (first (set->list piece)))
                 (pair? (second (first (set->list piece)))))
            ;; (v, (i, j)) => (v, (d-i, j))
            (set-map piece
                     (λ (x)
                       (list (first x)
                             (list (- d (first (second x)))
                                   (second (second x)))) ))
            ;; (i, j) => (d-i, j)
            (set-map piece
                     (λ (coord)
                       (list (- d (first coord))
                             (second coord))))))))

;; 获取上左角
(define (ulcorner patch)
  (define idxs (toindices patch))
  (list (apply min (map first (set->list idxs)))
        (apply min (map second (set->list idxs)))))

;; 获取下右角
(define (lrcorner patch)
  (define idxs (toindices patch))
  (list (apply max (map first (set->list idxs)))
        (apply max (map second (set->list idxs)))))

;; 计算对象的所有属性
(define (compute-object-properties element)
  (define props (make-hash))
  (hash-set! props 'size (size (toindices element)))
  (hash-set! props 'shape (shape element))
  (hash-set! props 'palette (palette element))
  (hash-set! props 'numcolors (numcolors element))
  (hash-set! props 'mostcolor (mostcolor element))
  (hash-set! props 'leastcolor (leastcolor element))
  props

// ...existing code...
