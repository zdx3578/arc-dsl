#lang rosette

(require racket/set
         racket/match)

;; ----------------------------------------------------------------------
;; 1. 数据结构定义与构造函数
;; ----------------------------------------------------------------------

;; 定义一个 Cell 结构
(struct Cell (value loc) #:transparent)

;; 定义一个 Object 结构，包含一组 Cells
(struct Object (cells) #:transparent)

;; 定义一个 Grid，使用嵌套列表表示
;; 例如：((1 2 2) (1 0 2) (3 3 3))
;; Grid 的索引从 0 开始
(struct Grid (rows) #:transparent)

;; ----------------------------------------------------------------------
;; 2. 辅助函数
;; ----------------------------------------------------------------------

;; 获取网格高度
(define (grid-height grid)
  (length (Grid-rows grid)))

;; 获取网格宽度（假设非空网格）
(define (grid-width grid)
  (length (first (Grid-rows grid))))

;; 获取 (i, j) 格子的颜色值
(define (grid-ref grid loc)
  (let ([i (first loc)]
        [j (second loc)])
    (list-ref (list-ref (Grid-rows grid) i) j)))

;; 计算网格中出现次数最多的颜色（作为背景色）
(define (mostcolor grid)
  (define freq (make-hash))
  (for*/list ([i (in-range (grid-height grid))]
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

;; ----------------------------------------------------------------------
;; 3. 邻居函数
;; ----------------------------------------------------------------------

;; 根据 diagonal? 参数返回相应的邻居
(define (neighbors loc h w diagonal?)
  (define i (first loc))
  (define j (second loc))
  (define base-neighs
    (list
     (list i     (add1 j))   ; 右
     (list i     (sub1 j))   ; 左
     (list (add1 i) j)       ; 下
     (list (sub1 i) j)))     ; 上
  (define diag-neighs
    (if diagonal?
        (list
         (list (add1 i) (add1 j)) ; 右下
         (list (add1 i) (sub1 j)) ; 左下
         (list (sub1 i) (add1 j)) ; 右上
         (list (sub1 i) (sub1 j))) ; 左上
        '()))
  (define all-neighs (append base-neighs diag-neighs))
  ;; 过滤越界
  (filter (λ (p)
            (define x (first p))
            (define y (second p))
            (and (>= x 0) (< x h) (>= y 0) (< y w)))
          all-neighs))

;; ----------------------------------------------------------------------
;; 4. BFS 实现
;; ----------------------------------------------------------------------

;; 从起点 loc 开始，扩展出一个对象
;; 如果 univalued? 为 #t，则对象内颜色必须 == start-color
;; 否则，只要不是背景色 bg 就可加入
(define (bfs-one-object grid start-loc start-color univalued? bg diagonal?)
  (define h (grid-height grid))
  (define w (grid-width grid))

  ;; 使用纯函数式 BFS，使用 tail recursion + visited set + queue list
  (define (loop queue visited acc)
    (cond
      [(null? queue)
       ;; BFS 完毕，返回 acc
       acc]
      [else
       (define cand (car queue))
       (define restq (cdr queue))
       (define cand-color (grid-ref grid cand))
       ;; 如果符合条件，则加入 acc
       (define acc2 (if (and (not univalued?)
                             (not (equal? cand-color bg)))
                        (set-add acc (Cell cand-color cand))
                        (if (and univalued?
                                 (equal? cand-color start-color))
                            (set-add acc (Cell cand-color cand))
                            acc)))
       ;; 找 cand 的邻居
       (define neighs (neighbors cand h w diagonal?))
       ;; 过滤：尚未访问、且符合颜色条件
       (define new-neighs
         (filter
          (λ (nloc)
            (and (not (set-member? visited nloc))
                 (if univalued?
                     (equal? (grid-ref grid nloc) start-color)
                     (not (equal? (grid-ref grid nloc) bg)))))
          neighs))
       ;; 标记 new-neighs 为已访问
       (define visited2 (foldl set-add visited new-neighs))
       ;; 入队
       (define queue2 (append restq new-neighs))
       ;; 递归调用
       (loop queue2 visited2 acc2)]))

  ;; 初始化
  (define visited0 (set start-loc))
  (define queue0 (list start-loc))
  (define acc0 (set)) ; 用 set 保存 Cells

  ;; 执行 BFS
  (loop queue0 visited0 acc0))

;; ----------------------------------------------------------------------
;; 5. 主函数：objects
;; ----------------------------------------------------------------------

;; 提取网格中的所有对象
;; 输入：
;; - grid: Grid
;; - univalued?: Boolean
;; - diagonal?: Boolean
;; - without-bg?: Boolean
;; 输出：
;; - Objects: Set of Object
(define (objects grid univalued? diagonal? without-bg?)
  (define h (grid-height grid))
  (define w (grid-width grid))

  ;; 计算背景颜色
  (define bg
    (if without-bg?
        (mostcolor grid)
        #f)) ; #f 表示无背景限制

  ;; 获取所有坐标
  (define all-locs
    (for*/list ([i (in-range h)]
                [j (in-range w)])
      (list i j)))

  ;; 定义一个递归函数，遍历所有坐标并提取对象
  (define (outer-loop locs visited objs)
    (cond
      [(null? locs)
       (set objs)] ; 返回一组对象的集合
      [else
       (define loc (car locs))
       (define rest-locs (cdr locs))
       (if (set-member? visited loc)
           ;; 已被某对象覆盖，跳过
           (outer-loop rest-locs visited objs)
           ;; 否则，检查是否为背景色
           (let ([col (grid-ref grid loc)])
             (if (and without-bg? (equal? col bg))
                 ;; 是背景色，标记为已访问并跳过
                 (outer-loop rest-locs (set-add visited loc) objs)
                 ;; 否则，启动 BFS 提取对象
                 (let ([obj-cells (bfs-one-object grid loc col univalued? bg diagonal?)])
                   ;; 更新 visited 集合
                   (define new-visited
                     (foldl (λ (cell s)
                              (set-add s (Cell-loc cell)))
                            visited
                            (set->list obj-cells)))
                   ;; 创建 Object 结构
                   (define new-object (Object obj-cells))
                   ;; 递归调用
                   (outer-loop rest-locs new-visited (cons new-object objs)))))]))

  ;; 执行外层循环
  (define objs (outer-loop all-locs (set) '()))
  objs)

;; 辅助函数：从 Cell 结构中提取位置
(define (Cell-loc cell)
  (Cell-loc cell))

;; ----------------------------------------------------------------------
;; 6. 翻转函数
;; ----------------------------------------------------------------------

;; 通用翻转函数
(define (mirror piece axis)
  (cond
    ;; 如果 piece 是 Grid，根据轴进行翻转
    [(Grid? piece)
     (cond
       [(eq? axis 'hmirror) ; 水平翻转：反转行顺序
        (Grid (reverse (Grid-rows piece)))]
       [(eq? axis 'vmirror) ; 垂直翻转：反转每一行中的元素
        (Grid (map reverse (Grid-rows piece)))]
       [(eq? axis 'dmirror) ; 对角线翻转：转置
        (Grid (transpose (Grid-rows piece)))]
       [(eq? axis 'cmirror) ; 反对角线翻转：先水平翻转，再对角线翻转
        (Grid (transpose (reverse (Grid-rows piece))))]
       [else (error "Unknown mirror axis")])]
    ;; 如果 piece 是 Object，根据轴进行翻转
    [(Object? piece)
     (define cells (Object-cells piece))
     (define flipped-cells
       (set (map
             (λ (cell)
               (define i (first (Cell-loc cell)))
               (define j (second (Cell-loc cell)))
               (cond
                 [(eq? axis 'hmirror)
                  (Cell (Cell-value cell) (list (- (grid-height grid) 1 i) j))]
                 [(eq? axis 'vmirror)
                  (Cell (Cell-value cell) (list i (- (grid-width grid) 1 j)))]
                 [(eq? axis 'dmirror)
                  (Cell (Cell-value cell) (list j i))]
                 [(eq? axis 'cmirror)
                  (Cell (Cell-value cell) (list (- (grid-width grid) 1 j) (- (grid-height grid) 1 i)))]
                 [else (error "Unknown mirror axis")]))
             (set->list cells))))
     (Object flipped-cells)]
    [else (error "Unknown Piece type")]))

;; 水平翻转
(define (hmirror piece)
  (mirror piece 'hmirror))

;; 垂直翻转
(define (vmirror piece)
  (mirror piece 'vmirror))

;; 对角线翻转
(define (dmirror piece)
  (mirror piece 'dmirror))

;; 反对角线翻转
(define (cmirror piece)
  (mirror piece 'cmirror))

;; ----------------------------------------------------------------------
;; 7. 对象属性计算（可选）
;; ----------------------------------------------------------------------

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

;; ----------------------------------------------------------------------
;; 8. 测试示例
;; ----------------------------------------------------------------------

;; 定义一个简单网格
(define grid1
  (Grid
   '((1 1 1 0)
     (1 0 1 0)
     (0 0 2 2)
     (3 3 3 0))))

;; 提取对象
(define my-objects
  (objects grid1 #t #f #t)) ;; univalued?=#t, diagonal?=#f, without_bg?=#t

;; 打印提取的对象
(displayln "Extracted Objects:")
(for ([obj (set->list my-objects)])
  (displayln (Object-cells obj)))

;; ----------------------------------------------------------------------
;; 9. 进一步集成与使用
;; ----------------------------------------------------------------------

;; 假设你想在提取对象后对其进行各种操作，如移动、镜像等，可以如下操作：

;; 示例：水平翻转所有对象
(define flipped-objects
  (set (map (λ (obj) (Object (map (λ (cell)
                                    (define i (first (Cell-loc cell)))
                                    (define j (second (Cell-loc cell)))
                                    (Cell (Cell-value cell) (list (- (grid-height grid1) 1 i) j)))
                                    )
                                  (Object-cells obj))))
              (set->list my-objects))))

(displayln "Flipped Objects (hmirror):")
(for ([obj (set->list flipped-objects)])
  (displayln (Object-cells obj)))

;; 计算并显示每个对象的属性
(displayln "\nObjects' Properties:")
(for ([obj (set->list my-objects)])
  (define size (object-size obj))
  (define center (object-center obj))
  (define colorcount (object-colorcount obj))
  (printf "Object: ~a\nSize: ~a\nCenter: ~a\nColor Count: ~a\n\n"
          (Object-cells obj) size center colorcount))

;; ----------------------------------------------------------------------
;; 10. 完整的函数与模块化（可选）
;; ----------------------------------------------------------------------

;; 如果需要将功能模块化，可以按功能划分不同的模块或文件：
;; - data-structures.rkt：定义结构体
;; - helpers.rkt：辅助函数
;; - bfs.rkt：BFS 实现
;; - mirror.rkt：翻转函数
;; - objects.rkt：主对象提取函数
;; - properties.rkt：对象属性计算
;; - main.rkt：主程序，集成以上模块并运行

;; 这里为了简洁性，将所有功能集成在一个文件中。

;; ----------------------------------------------------------------------
;; 11. 总结
;; ----------------------------------------------------------------------

; 1. **数据结构与辅助函数**：定义了 `Cell`、`Object`、`Grid` 结构体，并实现了辅助函数来操作和查询网格。

; 2. **邻居函数**：根据 `diagonal?` 参数选择 4 方向或 8 方向邻居，确保不越界。

; 3. **BFS 实现**：使用纯函数式的 BFS 方法，提取单个对象，并避免使用可变状态，适合 Rosette 的符号执行。

; 4. **主 `objects` 函数**：遍历所有格子，利用 BFS 提取所有对象，并返回一个对象集合。

; 5. **翻转函数**：实现了四种翻转操作，可以应用于 `Grid` 或 `Object`。

; 6. **对象属性计算**（可选）：为每个对象计算大小、中心位置和颜色统计等属性，便于后续分析或约束。

; 7. **测试示例**：提供了一个简单的网格示例，展示了如何提取对象、进行翻转操作以及计算对象属性。

; 8. **进一步集成与使用**：展示了如何对提取的对象进行各种操作和属性计算，确保流程的连贯性和可扩展性。

; 9. **模块化**（可选）：建议将不同功能分离到不同模块，提高代码的可维护性和复用性。

; 10. **性能考虑**：对于 30×30 的网格，采用纯函数式 BFS 的性能是足够的。只要避免在求解器中枚举大量可能的对象分割，整个流程可以高效运行。

; 希望这个完整的 Rosette/Racket 实现能满足你的需求，并为进一步的开发和优化提供一个坚实的基础。如有更多问题或需要进一步的功能扩展，欢迎随时提问！
