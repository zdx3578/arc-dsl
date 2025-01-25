;; data-structures.rkt
#lang rosette

(provide (struct-out Cell)
        ;;;  (struct-out Object)
         (struct-out Grid))

;; 定义一个 Cell 结构，包含颜色值和位置
(struct Cell (value loc) #:transparent)

;; 定义一个 Object 结构，包含一组 Cells
;;; (struct Object (cells) #:transparent)

;; 定义一个 Grid 结构，使用嵌套列表表示
(struct Grid (rows) #:transparent)

