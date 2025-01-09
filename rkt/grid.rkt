#lang rosette

(require racket/set
         racket/match)

;; ----------------------------------------------------------------------
;; 1. 辅助函数：获取网格宽高、访问格子、寻找背景色
;; ----------------------------------------------------------------------

;; 读取网格高度
(define (grid-height grid)
  (length grid))

;; 读取网格宽度（假设非空网格）
(define (grid-width grid)
  (length (first grid)))

;; 获取 (i, j) 格子的颜色值
;; 假设 grid 是一个 list-of-lists，i行j列
(define (grid-ref grid loc)
  (let ([i (first loc)]
        [j (second loc)])
    (list-ref (list-ref grid i) j)))

;; （示例）mostcolor：统计网格里出现次数最多的颜色，视为“背景”
;; 这里做一个简单频数统计，如果你有更优方法可替换
(define (mostcolor grid)
  (define freq (make-hash))
  (for* ([i (in-range (grid-height grid))]
         [j (in-range (grid-width grid))])
    (define c (grid-ref grid (list i j)))
    (hash-update! freq c (λ (old) (+ old 1)) 1))
  ;; 找出出现次数最多的 color
  (define-values (bg col-count)
    (argmax (hash->list freq) (λ (p) (second p))))
  bg)

;; 一个帮助函数，用于找最大值
(define (argmax lst val-fn)
  (foldl (λ (x best)
           (if (> (val-fn x) (val-fn best))
               x
               best))
         (first lst)
         (rest lst)))

;; ----------------------------------------------------------------------
;; 2. 邻居函数：根据 diagonal? 选择 4 连通或 8 连通
;; ----------------------------------------------------------------------

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
         (list (sub1 i) (sub1 j)) ; 左上
         )
        '()))
  (define all-neighs (append base-neighs diag-neighs))
  ;; 过滤越界
  (filter (λ (p)
            (define x (first p))
            (define y (second p))
            (and (>= x 0) (< x h) (>= y 0) (< y w)))
          all-neighs))

;; ----------------------------------------------------------------------
;; 3. BFS：给定起点 loc，扩展出一个对象
;;    - 如果 univalued?=#t，则对象内颜色必须 == start-color
;;    - 如果 univalued?=#f，则只要非背景就可加入
;; ----------------------------------------------------------------------

(define (bfs-one-object grid start-loc start-color univalued? bg diagonal?)
  (define h (grid-height grid))
  (define w (grid-width grid))

  ;; 我们做“纯函数式 BFS”，使用 tail recursion + visited set + queue list
  (define (loop queue visited acc)
    (cond
      [(null? queue)
       ;; BFS 完毕，返回 (visited, acc)
       (values visited acc)]
      [else
       (define cand (car queue))
       (define restq (cdr queue))
       (define cand-color (grid-ref grid cand))
       ;; 把 (cand-color, cand) 加到 acc
       (define new-acc (set-add acc (list cand-color cand)))

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
       ;; 把 new-neighs 标记 visited
       (define visited2 (foldl set-add visited new-neighs))
       ;; 入队
       (define queue2 (append restq new-neighs))
       (loop queue2 visited2 new-acc)]))

  (define visited0 (set start-loc))
  (define queue0 (list start-loc))
  (define acc0 (set)) ; 用 set 保存 (color (i j)) 组成

  (define-values (visited-final obj-final) (loop queue0 visited0 acc0))
  obj-final)

;; ----------------------------------------------------------------------
;; 4. 主函数：objects
;;    输入：grid, univalued?, diagonal?, without-bg?
;;    输出：一组对象，每个对象是 set-of (color (i j))
;; ----------------------------------------------------------------------

(provide objects)
(define (objects grid univalued? diagonal? without-bg?)
  (define h (grid-height grid))
  (define w (grid-width grid))

  ;; 背景颜色
  (define bg
    (if without-bg?
        (mostcolor grid)
        ;; 如果不排除背景，就设 #f 表示“无背景限制”
        #f))

  ;; 所有坐标
  (define all-locs
    (for*/list ([i (in-range h)]
                [j (in-range w)])
      (list i j)))

  ;; BFS 要用 visited 记录哪些格子已经归属某个对象
  ;; 我们用一个纯函数式 “outer loop” 来遍历 all-locs
  (define (outer-loop locs visited objs)
    (cond
      [(null? locs)
       (set objs)] ; 返回一组对象(列表)也可用 (set objs)
      [else
       (define loc (car locs))
       (define rest-locs (cdr locs))
       (if (set-member? visited loc)
           ;; 已在某对象内，跳过
           (outer-loop rest-locs visited objs)
           ;; 否则看看是否需要跳过背景
           (let ([col (grid-ref grid loc)])
             (if (equal? col bg)
                 ;; 是背景，跳过
                 (outer-loop rest-locs (set-add visited loc) objs)
                 ;; 启动 BFS
                 (define obj (bfs-one-object grid loc col univalued? bg diagonal?))
                 ;; BFS 返回的对象包含所有格子( (color,(i j)) )，我们只要把其中的 (i j) part
                 ;; or 直接保存( color, (i j) )看你的需求

                 ;; BFS 过程本身会把 visited locs, 但 outer 需要 union
                 (define new-visited
                   (foldl (λ (elt s)
                            (define loc2 (second elt))
                            (set-add s loc2))
                          visited
                          (set->list obj)))
                 (outer-loop rest-locs new-visited (cons obj objs)))))]))

  (define objs
    (outer-loop all-locs (set) '()))
  objs)

;; ----------------------------------------------------------------------
;; 5. 测试示例
;; ----------------------------------------------------------------------

;; 定义一个简单网格
(define grid1
  '((1 1 1 0)
    (1 0 1 0)
    (0 0 2 2)
    (3 3 3 0)))

;; 示例调用：提取对象
(define my-objects
  (objects grid1 #t #f #t)) ;; univalued?=#t, diagonal?=#f, without_bg?=#t

(displayln "Extracted Objects:")
(for ([obj (set->list my-objects)])
  (displayln obj))

