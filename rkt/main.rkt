#lang rosette

(require racket/set
         "objects.rkt"       ;; 假设里头有 (grid->objects grid), (rotate90 obj), (hmirror obj), etc.
         "data-structures.rkt") ;; 包含我们对 (struct Object ...) 的定义等

;;; 核心思路：

;;; 定义一个小 DSL，例如只允许 (rotate90 ...)、(hmirror ...)、(vmirror ...)、(no-op ...) 这样的组合；
;;; 用符号变量表示该 DSL 的抽象语法树（AST）；
;;; 写一个解释器 interp(e, obj)，能执行 DSL 的指令；
;;; 对每个训练例（input-grid => input-objects, output-objects），用 assert 约束 (interp symbolic-program input-objects) == output-objects;
;;; 调用 solve!，若可满足，就得到一种变换程序 symbolic-program；
;;; 后面如果要增加更多变换或判断特征(如「某对象满足某条件时执行 X，否则执行 Y」)，可以不断扩充 DSL 与解释器。

;; ----------------------------------------------------------------------------
;; 1. 简单的 DSL 定义：Expr 只能是下列几种变换之一
;;    - 'NoOp'        (什么都不做)
;;    - 'Rot90 e'     (先对 e 进行变换，再 rotate90)
;;    - 'HMirror e'   (先对 e 进行变换，再 hmirror)
;;    - 'VMirror e'   (先对 e 进行变换，再 vmirror)
;;    - 'Compose e1 e2' (先应用 e1，再应用 e2)
;;
;;   你可以再加 rotate180, dmirror, cmirror, color-shift...等等
;; ----------------------------------------------------------------------------

;; 首先定义一个 Rosette 数据类型(ADT)——用 define-syntax 或 define符号也行
;; 这里用 define-symbolic* + enumerations 只是示例。

;; --- 我们先写一个小结构，表示变换表达式 ---
;;   这里手写一个枚举 + 递归 AST 例子（非常简略）
(define-type (Expr)
  (U 'NoOp
     (Rot90 Expr)
     (HMirror Expr)
     (VMirror Expr)
     (Compose Expr Expr)))


;; ----------------------------------------------------------------------------
;; 2. 解释器：interp(e, obj) => 把 "DSL表达式 e" 作用到对象 obj 上
;; ----------------------------------------------------------------------------

(define (interp e obj)
  (match e
    ['NoOp
     obj]  ;; 不做任何变换

    [(Rot90 sub)
     (rotate90 (interp sub obj))]

    [(HMirror sub)
     (hmirror (interp sub obj))]

    [(VMirror sub)
     (vmirror (interp sub obj))]

    [(Compose e1 e2)
     (define r1 (interp e1 obj))
     (interp e2 r1)]))

;; ----------------------------------------------------------------------------
;; 3. 为了让 Rosette 能符号求解 "Expr"，
;;    我们需要 *符号化* 这个表达式 e
;; ----------------------------------------------------------------------------

;; 下面是个简单写法：我们限制 DSL 的深度到 2，或者 3。
;; 你可以写成 grammar-based enumerations 或 define-grammar DSL。
;; 此处只是一个 "硬编码" 例子: (define-symbolic expr1 expr2 expr3) + constraints 。
;;
;; 例：我们定义 "expr" = Compose e1 e2, e1 e2 都可以是 'NoOp', 'Rot90', 'HMirror', 'VMirror' 等
;;   也可以写成更复杂的递归 grammar.

(define-symbolic e1 e2 SymbolicAtomic)
;; e1,e2 ∈ { 'NoOp, 'Rot90, 'HMirror, 'VMirror } for example
(assert (or (eq? e1 'NoOp)
            (eq? e1 'Rot90)
            (eq? e1 'HMirror)
            (eq? e1 'VMirror)))

(assert (or (eq? e2 'NoOp)
            (eq? e2 'Rot90)
            (eq? e2 'HMirror)
            (eq? e2 'VMirror)))

;; 我们最终的 symbolic-expr = (Compose e1 e2)
;; 同时如果 e1='Rot90', interpret = (Rot90 'NoOp)? 不是很灵活
;; 这里仅做**示例**：你可以写 match 将 e1,e2 变成 DSL AST。
;;   => e1=NoOp => (Rot90 NoOp) 之类? 视需求而定。

(define (make-dsl e)
  (match e
    ['NoOp  'NoOp]
    ['Rot90 (Rot90 'NoOp)]   ;; 这里固定 sub = 'NoOp
    ['HMirror (HMirror 'NoOp)]
    ['VMirror (VMirror 'NoOp)]))

;; symbolic-expr 由 e1,e2 组合
(define (symbolic-expr e1 e2)
  (Compose (make-dsl e1) (make-dsl e2)))


;; ----------------------------------------------------------------------------
;; 4. 约束：对于每个 (input-grid => output-grid) 训练样例
;;    - 先把 input-grid => input-objects
;;    - symbolic-expr 作用到 input-objects => must = output-objects
;; ----------------------------------------------------------------------------

;; 假设我们有一批例子 ex-list: list of (cons input-grid output-grid)
;; 这里仅给**一个**例子为示范:
(define example1-in  (Grid '((0 0 0)
                             (7 7 7))))
(define example1-out (Grid '((7 7 7)
                             (0 0 0))))  ;; maybe want a hmirror

;; 先把grid转成对象, 只是演示
(define example1-in-objs  (objects example1-in #f #f #f))   ;; univalued?=#f, diag?=#f, w/o-bg?=#f
(define example1-out-objs (objects example1-out #f #f #f))

;; 这里**假设**我们只关心"单一对象"场景 => set里只有一个object
;; 真实ARC可能有多个对象。那就要对**每个对象**做变换并比对结果… 视需求而定。

(define example1-in-obj
  (first (set->list example1-in-objs)))    ;; 取出第一个 object
(define example1-out-obj
  (first (set->list example1-out-objs)))


;; 加入约束：interp( Compose e1 e2, example1-in-obj ) = example1-out-obj
(assert (equal? (interp (symbolic-expr e1 e2) example1-in-obj)
                example1-out-obj))

;; ----------------------------------------------------------------------------
;; 5. 求解！
;; ----------------------------------------------------------------------------
(define result (solve))
(if (sat? result)
    (begin
      (displayln "Solution found!")
      (define m (solution result))
      (displayln (format " e1=~a, e2=~a" (hash-ref m 'e1) (hash-ref m 'e2)))
      (displayln "Check final expression => ")
      (define final-expr (symbolic-expr (hash-ref m 'e1) (hash-ref m 'e2)))
      (displayln final-expr)
      (displayln "Test interpret => ")
      (displayln (interp final-expr example1-in-obj)))
    (displayln "No solution..."))


;; ----------------------------------------------------------------------------
;; 6. 后续扩展
;; ----------------------------------------------------------------------------
;; 1) 你可以把 DSL 改成更复杂的 grammar，用 "define-grammar" 或 define-lemma approach
;; 2) 你可以对多个例子 (train1..trainN) 做(for ([ex ex-list]) (assert ...)) => 让合成的同一表达式在全部例子上都成立
;; 3) 你提到“后面再继续增加其他判断函数、特征判断”等：是典型在 DSL 里加 if-then-else / match-color / shape-check 之类，
;;    并在 interp 里实现 => Rosette 就能符号执行并自动搜索满足全部例子的程序。
;; ----------------------------------------------------------------------------
