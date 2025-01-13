#lang rosette

;;; sonnet gen

(provide (all-defined-out))

;; 基础工具函数
(define (toindices obj)
  (if (set? obj)
      (if (list? (first (set->list obj)))
          (set-map obj (λ (x) (second x)))  ; 对于 Object
          obj)                              ; 对于 Indices
      (error "Invalid input type")))

;; 角点计算函数
(define (ulcorner patch)
  (define indices (toindices patch))
  (define coords (apply map list (set->list indices)))
  (list (apply min (first coords)) (apply min (second coords))))

(define (lrcorner patch)
  (define indices (toindices patch))
  (define coords (apply map list (set->list indices)))
  (list (apply max (first coords)) (apply max (second coords))))

;; 对象属性计算函数
(define (size obj)
  (set-count obj))

(define (shape obj)
  (define corners (map (λ (fn) (fn obj)) (list ulcorner lrcorner)))
  (define ul (first corners))
  (define lr (second corners))
  (list (add1 (- (first lr) (first ul)))
        (add1 (- (second lr) (second ul)))))

(define (palette element)
  (if (tuple? element)
      (list->set (flatten (tuple->list element)))
      (list->set (map first (set->list element)))))

(define (numcolors element)
  (set-count (palette element)))

(define (colorcount element value)
  (if (tuple? element)
      (length (filter (λ (x) (equal? x value)) (flatten (tuple->list element))))
      (length (filter (λ (x) (equal? (first x) value)) (set->list element)))))

(define (mostcolor element)
  (define vals
    (if (tuple? element)
        (flatten (tuple->list element))
        (map first (set->list element))))
  (define freqs (make-hash))
  (for ([v vals])
    (hash-set! freqs v (add1 (hash-ref freqs v 0))))
  (argmax (λ (k) (hash-ref freqs k)) (hash-keys freqs)))

(define (leastcolor element)
  (define vals
    (if (tuple? element)
        (flatten (tuple->list element))
        (map first (set->list element))))
  (define freqs (make-hash))
  (for ([v vals])
    (hash-set! freqs v (add1 (hash-ref freqs v 0))))
  (argmin (λ (k) (hash-ref freqs k)) (hash-keys freqs)))

;; 镜像和旋转函数
(define (hmirror piece)
  (if (tuple? piece)
      (reverse piece)
      (let ([d (+ (first (ulcorner piece)) (first (lrcorner piece)))])
        (if (list? (first (set->list piece)))
            (set-map piece (λ (x) (list (first x) (list (- d (first (second x))) (second (second x))))))
            (set-map piece (λ (x) (list (- d (first x)) (second x))))))))

;; 计算对象的所有属性
(define (compute-object-properties obj)
  (define props (make-hash))
  (hash-set! props 'size (size obj))
  (hash-set! props 'shape (shape obj))
  (hash-set! props 'palette (palette obj))
  (hash-set! props 'numcolors (numcolors obj))
  (when (not (set-empty? obj))
    (hash-set! props 'mostcolor (mostcolor obj))
    (hash-set! props 'leastcolor (leastcolor obj)))
  props)
