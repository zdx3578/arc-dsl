#lang rosette

(require racket/set
         racket/match)

;; ----------------------------
;; 数据结构定义与构造函数
;; ----------------------------

;; 定义一个 Cell
(define (make-cell v i j)
  (list v (list i j)))

;; 检查是否为 Cell
(define (cell? x)
  (and (list? x)
       (= (length x) 2)
       (integer? (first x))
       (list? (second x))
       (= (length (second x)) 2)
       (integer? (first (second x)))
       (integer? (second (second x)))))

;; 定义一个 Object
(define (make-object cells)
  (set cells))

;; 检查是否为 Object
(define (object? x)
  (and (set? x)
       (forall? cell? x)))

;; 定义一个 Grid
(define (make-grid rows)
  rows)

;; 检查是否为 Grid
(define (grid? x)
  (and (list? x)
       (forall? (λ (row) (and (list? row)
                               (forall? integer? row)))
                x)))

;; 定义一个 Piece
(define (make-piece grid-or-patch)
  grid-or-patch)

;; 检查是否为 Piece
(define (piece? x)
  (or (grid? x)
      (object? x))) ; 根据 asobject 函数，Patch 主要是 Object

;; ----------------------------
;; 辅助函数
;; ----------------------------

;; 获取左上角坐标 (最小的 i 和 j)
(define (ulcorner piece)
  (define coords (map (lambda (item)
                        (cond
                          [(cell? item) (second item)]
                          [else (error "Invalid Cell format")]))
                      piece))
  (define min-i (apply min (map first coords)))
  (define min-j (apply min (map second coords)))
  (list min-i min-j))

;; 获取右下角坐标 (最大的 i 和 j)
(define (lrcorner piece)
  (define coords (map (lambda (item)
                        (cond
                          [(cell? item) (second item)]
                          [else (error "Invalid Cell format")]))
                      piece))
  (define max-i (apply max (map first coords)))
  (define max-j (apply max (map second coords)))
  (list max-i max-j))

;; ----------------------------
;; asobject 函数实现
;; ----------------------------

(define (asobject grid)
  (unless (grid? grid)
    (error "Input must be a Grid"))
  (define obj
    (set (for*/list ([i (in-naturals)]
                     [row grid]
                     [j (in-naturals)]
                     [v row])
           (make-cell v i j))))
  obj)

;; ----------------------------
;; hmirror 函数实现
;; ----------------------------

(define (hmirror piece)
  (cond
    ;; 如果 piece 是 Grid，返回反转的 Grid
    [(grid? piece) (reverse piece)]
    ;; 如果 piece 是 Object
    [(object? piece)
     (let* ((d (+ (first (ulcorner piece))
                  (first (lrcorner piece))))
            ;; 计算新的坐标
            (mirrored-cells
             (map (lambda (cell)
                    (match cell
                      [(list v (list i j))
                       (make-cell v (- d i) j)]
                      [_ (error "Unexpected Cell structure")]))
                  (set->list piece))))
       ;; 创建新的 Object
       (make-object mirrored-cells))]
    [else (error "Unknown Piece type")]))

;; ----------------------------
;; 声明符号化的 Grid
;; ----------------------------

;; 假设我们希望符号化一个 Grid，并将其转换为 Object
(define-symbolic symbolic-grid (grid?))

;; 使用 asobject 函数将 Grid 转换为 Object
(define symbolic-object (asobject symbolic-grid))

;; 应用 hmirror 函数
(define mirrored-object (hmirror symbolic-object))

;; ----------------------------
;; 设定约束
;; ----------------------------

;; 示例约束：镜像后的 Object 中存在一个 Cell，其值为 2 且坐标为 (1, 1)
(assert (ormap (λ (cell)
                (and (= (first cell) 2)
                     (= (second cell) '(1 1))))
              (set->list mirrored-object)))

;; 另一个约束示例：确保镜像后的 Object 中所有的 i 坐标都在 [0,1]
(assert (forall (λ (cell)
                (let ((i (first (second cell))))
                  (and (>= i 0) (<= i 1))))
              mirrored-object)))

;; ----------------------------
;; 求解
;; ----------------------------

(define sol (solve))

;; 打印结果
(print sol)
