#lang racket

(provide base-grid target-grid rotate-left mirror-rows apply-op)
(require json)

; 从JSON文件读取网格数据
(define (read-grid-from-file path)
  (let* ([json-string (file->string path)]
         [json-data (string->jsexpr json-string)])
    (hash-ref json-data 'grid)))

; 修改原有定义
(define base-grid
  (read-grid-from-file "input/base-grid.json"))

(define target-grid
  (read-grid-from-file "input/target-grid.json"))

;; 镜像操作：按行反转
(define (mirror-rows grid)
  (map reverse grid))

;; 旋转操作：左旋90度（针对 2×2 网格）
(define (rotate-left grid)
  (list (list (second (first grid))
              (second (second grid)))
        (list (first  (first grid))
              (first  (second grid)))))

;; 应用操作
(define (apply-op grid op)
  (case op
    [(M) (mirror-rows grid)]
    [(R) (rotate-left grid)]
    [else (error "Unknown operation" op)]))
