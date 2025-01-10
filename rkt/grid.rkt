;; data-structures.rkt
#lang rosette

(provide Cell Object Grid)

;; 定义一个 Cell 结构，包含颜色值和位置
(struct Cell (value loc) #:transparent)

;; 定义一个 Object 结构，包含一组 Cells
(struct Object (cells) #:transparent)

;; 定义一个 Grid 结构，使用嵌套列表表示
(struct Grid (rows) #:transparent)



;; helpers.rkt
#lang rosette

(require "data-structures.rkt")

(provide grid-height grid-width grid-ref mostcolor argmax transpose)

;; 获取网格高度
(define (grid-height grid)
  (length (Grid-rows grid)))

;; 获取网格宽度（假设网格非空）
(define (grid-width grid)
  (length (first (Grid-rows grid))))

;; 获取 (i, j) 位置的颜色值
(define (grid-ref grid loc)
  (let ([i (first loc)]
        [j (second loc)])
    (list-ref (list-ref (Grid-rows grid) i) j)))

;; 计算网格中出现次数最多的颜色（作为背景色）
(define (mostcolor grid)
  (define freq (make-hash))
  (for ([i (in-range (grid-height grid))]
        [j (in-range (grid-width grid))])
    (define c (grid-ref grid (list i j)))
    (hash-update! freq c (λ (old) (+ old 1)) 1))
  ;; 找出出现次数最多的 color
  (define-values (bg _)
    (argmax (hash->list freq) (λ (p) (second p))))
  bg)

;; 找到列表中最大值元素的函数
(define (argmax lst val-fn)
  (foldl (λ (x best)
           (if (> (val-fn x) (val-fn best))
               x
               best))
         (first lst)
         (rest lst)))

;; 转置函数
(define (transpose grid)
  (apply map list grid))



;; bfs.rkt
#lang rosette

(require "data-structures.rkt"
         "helpers.rkt")

(provide bfs-one-object)

;; BFS: 从 start-loc 开始，提取一个对象
;; 参数：
;; - grid: Grid
;; - start-loc: (list i j)
;; - start-color: 颜色值
;; - univalued?: Boolean
;; - bg: 背景颜色（或 #f）
;; - diagonal?: Boolean
(define (bfs-one-object grid start-loc start-color univalued? bg diagonal?)
  (define h (grid-height grid))
  (define w (grid-width grid))

  ;; 纯函数式 BFS，使用递归
  (define (loop queue visited acc)
    (cond
      [(null? queue)
       acc]
      [else
       (define cand (car queue))
       (define restq (cdr queue))
       (define cand-color (grid-ref grid cand))
       ;; 判断是否将此单元格加入 acc
       (define acc2
         (cond
           [(and (not univalued?)
                 (not (equal? cand-color bg)))
            (set-add acc (Cell cand-color cand))]
           [(and univalued?
                 (equal? cand-color start-color))
            (set-add acc (Cell cand-color cand))]
           [else
            acc]))

       ;; 获取候选单元格的邻居
       (define neighs (neighbors cand h w diagonal?))

       ;; 过滤符合条件且未访问的邻居
       (define new-neighs
         (filter
          (λ (nloc)
            (and (not (set-member? visited nloc))
                 (if univalued?
                     (equal? (grid-ref grid nloc) start-color)
                     (not (equal? (grid-ref grid nloc) bg)))))
          neighs))

       ;; 更新已访问集合
       (define visited2 (foldl set-add visited new-neighs))

       ;; 更新队列
       (define queue2 (append restq new-neighs))

       ;; 递归调用
       (loop queue2 visited2 acc2)]))

  ;; 初始化已访问和队列
  (define visited0 (set start-loc))
  (define queue0 (list start-loc))
  (define acc0 (set))

  ;; 运行 BFS
  (loop queue0 visited0 acc0))




;; objects.rkt
#lang rosette

(require "data-structures.rkt"
         "helpers.rkt"
         "bfs.rkt")

(provide objects)

;; 提取网格中的所有对象
;; 参数：
;; - grid: Grid
;; - univalued?: Boolean
;; - diagonal?: Boolean
;; - without-bg?: Boolean
;; 返回：
;; - Set of Objects
(define (objects grid univalued? diagonal? without-bg?)
  (define h (grid-height grid))
  (define w (grid-width grid))

  ;; 确定背景颜色
  (define bg
    (if without-bg?
        (mostcolor grid)
        #f)) ; #f 表示不排除任何颜色

  ;; 获取所有坐标
  (define all-locs
    (for*/list ([i (in-range h)]
                [j (in-range w)])
      (list i j)))

  ;; 定义一个辅助函数，从 Cell 提取位置
  (define (cell-loc cell)
    (Cell-loc cell))

  ;; 外层循环：遍历所有坐标，提取对象
  (define (outer-loop locs visited objs)
    (cond
      [(null? locs)
       (set objs)] ; 返回对象集合
      [else
       (define loc (car locs))
       (define rest-locs (cdr locs))
       (if (set-member? visited loc)
           ;; 已经属于某个对象，跳过
           (outer-loop rest-locs visited objs)
           ;; 否则，检查是否为背景色
           (let ([col (grid-ref grid loc)])
             (if (and without-bg? (equal? col bg))
                 ;; 是背景色，标记为已访问并跳过
                 (outer-loop rest-locs (set-add visited loc) objs)
                 ;; 否则，启动 BFS 提取对象
                 (let ([obj-cells (bfs-one-object grid loc col univalued? bg diagonal?)])
                   ;; 更新已访问集合
                   (define new-visited
                     (foldl (λ (cell s)
                              (set-add s (cell-loc cell)))
                            visited
                            (set->list obj-cells)))
                   ;; 创建 Object 结构
                   (define new-object (Object obj-cells))
                   ;; 添加到对象集合并继续
                   (outer-loop rest-locs new-visited (cons new-object objs))))))]))

  ;; 执行外层循环
  (define objs (outer-loop all-locs (set) '()))
  objs)




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




;; properties.rkt
#lang rosette

(require "data-structures.rkt")

(provide object-size object-center object-colorcount)

;; 计算对象的大小（格子数）
(define (object-size obj)
  (set-size (Object-cells obj)))

;; 计算对象的中心位置（平均坐标）
(define (object-center obj)
  (define cells (set->list (Object-cells obj)))
  (define total-i (apply + (map (λ (cell) (first (Cell-loc cell))) cells)))
  (define total-j (apply + (map (λ (cell) (second (Cell-loc cell))) cells)))
  (define n (length cells))
  (list (/ total-i n) (/ total-j n)))

;; 计算对象的颜色统计（颜色 -> 数量）
(define (object-colorcount obj)
  (define cells (set->list (Object-cells obj)))
  (define freq (make-hash))
  (for ([cell cells])
    (define c (Cell-value cell))
    (hash-update! freq c (λ (old) (+ old 1)) 1))
  (hash->list freq))




;; json-reader.rkt
#lang rosette

(require "data-structures.rkt"
         json
         racket/file)

(provide read-json-file read-all-json-files)

;; 读取并解析单个 JSON 文件
;; 参数：
;; - filepath: string
;; 返回：
;; - 解析后的 JSON 数据（哈希表等）
(define (read-json-file filepath)
  "读取指定路径的 JSON 文件并解析为 Racket 数据结构。"
  (define json-str (file->string filepath))
  (read-json json-str))

;; 遍历目录并读取所有 JSON 文件
;; 参数：
;; - dir-path: string（目录路径）
;; 返回：
;; - 解析后的 JSON 数据列表
(define (read-all-json-files dir-path)
  "遍历指定目录，读取所有 JSON 文件并返回解析后的数据列表。"
  (define files (directory-list dir-path))
  (define json-files
    (filter (λ (f) (string-suffix? ".json" f)) files))
  (map read-json-file json-files))




;; process-data.rkt
#lang rosette

(require "data-structures.rkt"
         "helpers.rkt"
         "objects.rkt"
         "properties.rkt")

(provide process-json-data)

;; 处理单个 JSON 数据，提取对象并计算属性
;; 参数：
;; - json-data: 哈希表，包含 "train" 和 "test" 键
;; 返回：
;; - 无（打印输出）
(define (process-json-data json-data)
  "处理单个 JSON 数据，提取对象并执行验证。"

  ;; 提取训练和测试数据
  (define train-data (hash-ref json-data 'train))
  (define test-data (hash-ref json-data 'test))

  ;; 定义所有8种参数组合
  (define param-combinations
    (list
     (list #t #t #t)
     (list #t #t #f)
     (list #t #f #t)
     (list #t #f #f)
     (list #f #t #t)
     (list #f #t #f)
     (list #f #f #t)
     (list #f #f #f)))

  ;; 处理每个输入输出对
  (define (handle-data-pair data-pair)
    ;; 将输入和输出网格转换为 Grid 结构
    (define input-grid (Grid (map list (hash-ref data-pair 'input))))
    (define output-grid (Grid (map list (hash-ref data-pair 'output))))

    ;; 遍历所有参数组合
    (for ([params param-combinations])
      (define univalued? (first params))
      (define diagonal? (second params))
      (define without-bg? (third params))

      ;; 提取输入和输出对象
      (define input-objects (objects input-grid univalued? diagonal? without-bg?))
      (define output-objects (objects output-grid univalued? diagonal? without-bg?))

      ;; 计算输入对象的属性
      (define input-props
        (map (λ (obj)
               (hash
                'size (object-size obj)
                'center (object-center obj)
                'colorcount (object-colorcount obj)))
             (set->list input-objects)))

      ;; 计算输出对象的属性
      (define output-props
        (map (λ (obj)
               (hash
                'size (object-size obj)
                'center (object-center obj)
                'colorcount (object-colorcount obj)))
             (set->list output-objects)))

      ;; 打印参数组合
      (printf "Parameters: univalued?=~a, diagonal?=~a, without_bg?=~a\n"
              univalued? diagonal? without-bg?)

      ;; 打印输入对象
      (printf "Input Objects:\n")
      (for ([obj (set->list input-objects)])
        (displayln (Object-cells obj)))

      ;; 打印输入属性
      (printf "Input Properties:\n")
      (for ([prop input-props])
        (displayln prop))

      ;; 打印输出对象
      (printf "Output Objects:\n")
      (for ([obj (set->list output-objects)])
        (displayln (Object-cells obj)))

      ;; 打印输出属性
      (printf "Output Properties:\n")
      (for ([prop output-props])
        (displayln prop))

      ;; 分隔符
      (printf "----------------------------------------\n"))))

  ;; 处理所有训练数据
  (printf "Processing Train Data:\n")
  (for ([data-pair train-data])
    (handle-data-pair data-pair))

  ;; 处理所有测试数据
  (printf "Processing Test Data:\n")
  (for ([data-pair test-data])
    (handle-data-pair data-pair)))




;; main.rkt
#lang rosette

(require "json-reader.rkt"
         "process-data.rkt")

(provide main)

;; 主程序
(define (main)
  "主程序：读取目录下所有 JSON 文件并处理。"

  (define dir-path "./training-data") ; 替换为你的 JSON 文件目录路径
  (define all-json-data (read-all-json-files dir-path))

  ;; 循环处理每个 JSON 数据文件
  (for ([json-data all-json-data])
    (process-json-data json-data)))

;; 执行主程序
(main)




