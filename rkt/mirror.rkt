;; mirror.rkt
#lang rosette

(require "data-structures.rkt"
         "helpers.rkt")

(provide mirror hmirror vmirror dmirror cmirror)

;; 通用翻转函数
;; 参数：
;; - piece: Grid 或 Object
;; - axis: 'hmirror, 'vmirror, 'dmirror, 'cmirror
;; - grid: Grid (用于 Object 翻转以获取网格尺寸)
;; 返回：
;; - 翻转后的 piece
(define (mirror piece axis grid)
  (cond
    [(Grid? piece)
     (cond
       [(eq? axis 'hmirror)
        ;; 水平翻转：反转行顺序
        (Grid (reverse (Grid-rows piece)))]
       [(eq? axis 'vmirror)
        ;; 垂直翻转：反转每一行
        (Grid (map reverse (Grid-rows piece)))]
       [(eq? axis 'dmirror)
        ;; 对角线翻转：转置
        (Grid (transpose (Grid-rows piece)))]
       [(eq? axis 'cmirror)
        ;; 反对角线翻转：先水平翻转，再转置
        (Grid (transpose (reverse (Grid-rows piece))))]
       [else
        (error "Unknown mirror axis")])]

    [(Object? piece)
     (define cells (Object-cells piece))
     (define flipped-cells
       (set (map
             (λ (cell)
               (define i (first (Cell-loc cell)))
               (define j (second (Cell-loc cell)))
               (cond
                 [(eq? axis 'hmirror)
                  ;; 水平翻转：翻转 i 坐标
                  (Cell (Cell-value cell) (list (- (grid-height grid) 1 i) j))]
                 [(eq? axis 'vmirror)
                  ;; 垂直翻转：翻转 j 坐标
                  (Cell (Cell-value cell) (list i (- (grid-width grid) 1 j)))]
                 [(eq? axis 'dmirror)
                  ;; 对角线翻转：交换 i 和 j
                  (Cell (Cell-value cell) (list j i))]
                 [(eq? axis 'cmirror)
                  ;; 反对角线翻转：翻转 i 和 j 后交换
                  (Cell (Cell-value cell) (list (- (grid-width grid) 1 j) (- (grid-height grid) 1 i)))]
                 [else
                  (error "Unknown mirror axis")]))
             (set->list cells))))
     (Object flipped-cells)]

    [else
     (error "Unknown Piece type")]))

;; 定义具体的镜像函数，需要传入 grid 以便翻转 Objects
(define (hmirror piece grid)
  (mirror piece 'hmirror grid))

(define (vmirror piece grid)
  (mirror piece 'vmirror grid))

(define (dmirror piece grid)
  (mirror piece 'dmirror grid))

(define (cmirror piece grid)
  (mirror piece 'cmirror grid))
