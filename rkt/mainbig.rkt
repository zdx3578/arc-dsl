#lang rosette

;; -------------------------------------------------------
;; 0) 引入我们需要的模块
;; -------------------------------------------------------
(require racket/set
        ;;;  typed/racket
         rosette/lib/match
         "objects.rkt"
         "properties.rkt"
        ;;;  "typed-part.rkt"     ;; 假设里头有 (objects grid ...) (rotate90 obj), (hmirror obj), (vmirror obj) 等
         "json-reader.rkt"       ;; 提供 (read-all-json-files dir)
         "data-structures.rkt")  ;; 包含我们对 (struct Grid ...) 以及 (struct Object ...) 的定义等


;; -------------------------------------------------------
;; 1) DSL 定义（使用 struct 而非 define-type）
;;    参考第3版的写法，但我们在此扩展为支持: NoOp, Rot90, HMirror, VMirror, Compose
;; -------------------------------------------------------
(struct NoOp ()          #:transparent)
(struct Rot90 (sub)      #:transparent)
(struct HMirror (sub)    #:transparent)
(struct VMirror (sub)    #:transparent)
(struct Compose (e1 e2)  #:transparent)

;; 对应的底层函数 rotate90/hmirror/vmirror 等，请在 objects.rkt 或其他文件中实现
;; 这里暂时示例放一个空壳:
;; (define (rotate90 obj) ...)
;; (define (hmirror obj) ...)
;; (define (vmirror obj) ...)

;; -------------------------------------------------------
;; 2) 解释器：interp(e, obj) => 把 DSL 表达式作用到对象 obj 上
;; -------------------------------------------------------
(define (interp expr obj)
  (match expr
    [(NoOp)
     obj]
    [(Rot90 sub)
     (rotate90 (interp sub obj))]
    [(HMirror sub)
     (hmirror (interp sub obj))]
    [(VMirror sub)
     (vmirror (interp sub obj))]
    [(Compose e1 e2)
     (let ([r1 (interp e1 obj)])
       (interp e2 r1))]))

;; -------------------------------------------------------
;; 3) 符号化 DSL，并对其施加约束
;;    - 这里演示“深度不大”的简易符号化，把 e1,e2 ∈ {NoOp,Rot90,HMirror,VMirror}，
;;      然后做一个 (Compose (make-dsl e1) (make-dsl e2)) 。
;;
;;    你也可以做更灵活的递归定义/枚举，或 grammar-based approach。
;; -------------------------------------------------------

;;; (module typed-part typed/racket
;;;   (provide SymbolicAtomic)
;;;   (define-type SymbolicAtomic
;;;     (U 'NoOp 'Rot90 'HMirror 'VMirror)) )
;;; (require (submod "." typed-part))
;;; ;;; ;; 定义符号变量 e1,e2
;;; (define-symbolic e1 e2 SymbolicAtomic)

;;; (define-symbolic* e1 e2 (enum 'NoOp 'Rot90 'HMirror 'VMirror))




(define-symbolic e1 e2 symbol?)
;; 约束 e1,e2 只能是符号 'NoOp, 'Rot90, 'HMirror, 'VMirror
(assert (or (eq? e1 'NoOp)
            (eq? e1 'Rot90)
            (eq? e1 'HMirror)
            (eq? e1 'VMirror)))
(assert (or (eq? e2 'NoOp)
            (eq? e2 'Rot90)
            (eq? e2 'HMirror)
            (eq? e2 'VMirror)))

;; 把符号 'NoOp / 'Rot90 / 'HMirror / 'VMirror => 我们的 struct DSL
(define (make-dsl sym)
  (match sym
    ['NoOp    (NoOp)]
    ['Rot90   (Rot90 (NoOp))]
    ['HMirror (HMirror (NoOp))]
    ['VMirror (VMirror (NoOp))]
    [_        (error "Unexpected symbol for DSL" sym)]))

;; 把 e1,e2 组成一个Compose
(define (symbolic-expr e1 e2)
  (Compose (make-dsl e1)
           (make-dsl e2)))


           

;;; ;; -------------------------------------------------------
;;; ;; 4) 简单示例：对单个例子加约束 => 调用 solve
;;; ;;    （此处演示，只有一个例子 example1）
;;; ;; -------------------------------------------------------

;;; ;; 假设我们有 (struct Grid ...) 已在 data-structures.rkt 定义
;;; ;; 假设 (objects grid #f #f #f) 可将 Grid 转成 set-of-Object
;;; ;; 用它来示例一个简单的约束：
;;; (define example1-in  (Grid '((0 0 0)
;;;                              (7 7 7))))
;;; (define example1-out (Grid '((7 7 7)
;;;                              (0 0 0))))  ;; 可能是 hmirror 的结果

;;; (define example1-in-objs  (objects example1-in  #f #f #f))
;;; (define example1-out-objs (objects example1-out #f #f #f))

;;; ;; 为简化，假设只有一个object
;;; (define example1-in-obj
;;;   (first (set->list example1-in-objs)))
;;; (define example1-out-obj
;;;   (first (set->list example1-out-objs)))

;;; ;; 加入约束： interp((Compose e1 e2), input-obj) == output-obj
;;; (define constraints
;;;   (assert (equal? (interp (symbolic-expr e1 e2) example1-in-obj)
;;;                 example1-out-obj)))

;;; ;; -------------------------------------------------------
;;; ;; 5) 调用 solve
;;; ;; -------------------------------------------------------
;;; (define result (solve constraints))
;;; (if (sat? result)
;;;     (begin
;;;       (displayln "Solution found for example1!")
;;;       (define m (solution result))
;;;       (define found-e1 (hash-ref m 'e1))
;;;       (define found-e2 (hash-ref m 'e2))
;;;       (displayln (format " => e1=~a, e2=~a" found-e1 found-e2))

;;;       (displayln "Check final expression => ")
;;;       (define final-expr (symbolic-expr found-e1 found-e2))
;;;       (displayln final-expr)

;;;       (displayln "Test interpret => ")
;;;       (displayln (interp final-expr example1-in-obj)))
;;;     (displayln "No solution for example1..."))


;; -------------------------------------------------------
;; 6) 额外示例：加入第二版的 JSON 读取逻辑
;;    - 这里给一个 main 函数：读取某个目录下所有 JSON，
;;      然后对其中的第一个 train-pair，进行“对象提取 + 符号合成”的示例。
;;    - 你可以在命令行中执行:
;;         racket your-file.rkt <dir-of-json>
;; -------------------------------------------------------

;; 6.1) 一个帮助函数：对某个 input-obj => output-obj，求解 DSL 变换
(define (synthesize-transformation input-obj output-obj)
  ;; 定义一个新的符号变量 e
  ;; 这里示范另外一种做法：而不是 e1,e2，直接“整个 e”符号化
  ;;; (define-symbolic e (NoOp Rot90 HMirror VMirror Compose))
  ;; 上面这一行写法，仅仅是示例。实际可用 grammar-based approach，也可参照上面 e1,e2 的方式

  ;; 约束
  (define constraints
    (assert (equal? (interp e1 input-obj) output-obj)))

  (define r (solve constraints))
  (if r
    (begin
      (displayln "Solution found!")
      (define m (model r))   ; Rosette 2.x 风格
      (displayln m))
    (displayln "No solution...")))
  ;;; (if (sat? r)
  ;;;     (let ([sol (solution r)]
  ;;;           [found-e (lookup (solution r) e)])
  ;;;       (displayln "Found a transformation for this pair!")
  ;;;       (displayln (format " => DSL expr = ~a" found-e))
  ;;;       found-e)
  ;;;     (begin
  ;;;       (displayln "No solution found for this pair...")
  ;;;       #f)))

;; 6.2) 主函数：读取目录 => 对 JSON => 取出第一个 train pair => 合成
(define (main dir)
  (define all-json (read-all-json-files dir))  ;; => list of JSON data
  (for ([json-data (in-list all-json)])
    (displayln "======================================")
    (displayln (format "Now process JSON: ~a" json-data))

    (define train-data (hash-ref json-data 'train))
    (define test-data  (hash-ref json-data 'test))

    ;; 只示范：拿第一个 train pair
    (define first-pair (if (null? train-data) #f (car train-data)))
    (when first-pair
      (define input-grid  (Grid (hash-ref first-pair 'input)))
      (define output-grid (Grid (hash-ref first-pair 'output)))

      (define input-objects  (objects input-grid  #f #f #f))
      (define output-objects (objects output-grid #f #f #f))

      (define in-obj  (if (set-empty? input-objects)
                          (set)
                          (car (set->list input-objects))))
      (define out-obj (if (set-empty? output-objects)
                          (set)
                          (car (set->list output-objects))))

      (displayln (format "Input obj=~a" in-obj))
      (displayln (format "Output obj=~a" out-obj))

      ;; 调用我们写的合成函数
      (synthesize-transformation in-obj out-obj))))

;; -------------------------------------------------------
;; 7) 程序入口
;;    在命令行运行：
;;      racket this-file.rkt <dir>
;;    Rosette 里也可用 (main "某个路径") 方式直接调用
;; -------------------------------------------------------
(provide main)


;;; 说明
;;; DSL 定义方式

;;; 我们不再使用 (define-type (Expr) ...)，而是使用了 struct NoOp ()、struct Rot90 (sub) 等形式（参考第3版）。
;;; 通过 #:transparent 让结构体在打印时显示内部信息，便于调试。
;;; 解释器 interp

;;; 依旧采用模式匹配 (match expr ...) 的方式，对不同 DSL 分支进行相应的变换。
;;; 符号化 DSL

;;; 在 section 3 里，演示了类似第一版的做法：用 e1,e2 这样的符号变量，然后用 (Compose (make-dsl e1) (make-dsl e2)) 的方式来生成一个组合的 DSL 表达式。
;;; 在 section 6.1 的 synthesize-transformation 函数里，又示范了另一种可能：直接让 e 做成一个包含所有 struct 的通用符号变量（不过这种用法需要更多配置，一般要做 grammar-based approach 或在 struct 上加 #:mutable + define-symbolic* 才会更灵活）。
;;; 你可以根据需求选用其中一种方式，也可以扩展到更深的递归/嵌套。
;;; JSON 读取

;;; 在 section 6 中，引入了第2版的做法：写了一个 main 函数，读取指定目录下的所有 JSON (read-all-json-files dir)，并对每个 JSON 的第一个训练数据做一次“生成变换程序”的过程。
;;; 实际需求中，你可能会对每个 train-data 的多个 pair 做多次约束，从而逼近一个通用的 DSL 程序；也可能需要把 test-data 拿来检验。此处只是一个示例框架。
;;; 多对象/多例子场景

;;; 如果你的任务中需要对“多个对象”或“多个训练对”同时求解，可以在加约束时，把 (interp e obj) 对应所有对象都做变换，并与目标对象集做比较，或分配多个 DSL 表达式。
;;; 同理，如果 DSL 需要有更多分支（例如 IfColorThen e1 Else e2），也可以照此模式在 struct 里加新的分支，再在 interp 里写对应逻辑即可。
;;; 求解过程

;;; 第一个大示例（section 4-5）是直接在文件加载时就 (solve) 了，对 example1 做搜索。
;;; 第二个示例（section 6.2）则是等 main 被调用时，才会去做合成。
;;; 通常在 Rosette 项目里，你会统一在某个函数中先构建完所有约束，然后调用 solve 一次，得到解。上面只是演示多种写法。
;;; 通过以上示例，你就可以得到一个既包含第一版的“细节和多变换 DSL、组合约束”，又包含第二版的“JSON 读取流程”，同时还采用了第三版的 struct DSL 写法的综合示例。希望能帮到你更好地在实际项目中使用 Rosette 做程序合成与分析。






