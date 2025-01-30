#lang rosette

(require rosette/lib/match
         racket/hash
        ;;;  racket/list
         "objects.rkt"
         "properties.rkt"
         "json-reader.rkt"
         "data-structures.rkt")

;; -----------------------------------------------------------
;; 1) transformations + apply-op
;; -----------------------------------------------------------
(struct TransformationInfo (name code apply-fn check-fn dsl-maker) #:transparent)

(define transformations
  (list
   (TransformationInfo
    'NoOp
    0
    (lambda (obj) obj)
    (lambda (i o) (equal? i o))
    (lambda (sub) (NoOp))) ;; sub => #f

   (TransformationInfo
    'Rot90
    1
    (lambda (obj) (if (valid-rot90? obj) (rotate90 obj) #f))
    (lambda (i o) (equal? (rotate90 i) o))
    (lambda (sub) (Rot90 sub)))

   (TransformationInfo
    'HMirror
    2
    (lambda (obj) (if (valid-hmirror? obj) (hmirror obj) #f))
    (lambda (i o) (equal? (hmirror i) o))
    (lambda (sub) (HMirror sub)))

   (TransformationInfo
    'VMirror
    3
    (lambda (obj) (if (valid-vmirror? obj) (vmirror obj) #f))
    (lambda (i o) (equal? (vmirror i) o))
    (lambda (sub) (VMirror sub)))

   (TransformationInfo
    'CMirror
    4
    (lambda (obj) (cmirror obj))
    (lambda (i o) (equal? (cmirror i) o))
    (lambda (sub) (CMirror sub)))

   (TransformationInfo
    'DMirror
    5
    (lambda (obj) (dmirror obj))
    (lambda (i o) (equal? (dmirror i) o))
    (lambda (sub) (DMirror sub)))
   ))

;; 查找 transformation
(define (lookup-trans-by-code c)
  (for/first ([tf (in-list transformations)])
    (when (= c (TransformationInfo-code tf))
      tf)))

(define (lookup-trans-by-name nm)
  (for/first ([tf (in-list transformations)])
    (when (eq? nm (TransformationInfo-name tf))
      tf)))

;; 统一 apply-op: code => transformations
(define (apply-op code obj)
  (define tf (lookup-trans-by-code code))
  (if tf
      ((TransformationInfo-apply-fn tf) obj)
      #f))

;; -----------------------------------------------------------
;; 2) DSL 定义 & interp (含 Compose)
;; -----------------------------------------------------------
;; 与之前一样的 DSL (单操作 + Compose):
(struct DSL (op sub) #:transparent)

(define-syntax-rule (NoOp)
  (DSL 'NoOp #f))

(define-syntax-rule (Rot90 sub)
  (DSL 'Rot90 sub))

(define-syntax-rule (HMirror sub)
  (DSL 'HMirror sub))

(define-syntax-rule (VMirror sub)
  (DSL 'VMirror sub))

(define-syntax-rule (CMirror sub)
  (DSL 'CMirror sub))

(define-syntax-rule (DMirror sub)
  (DSL 'DMirror sub))

(define-syntax-rule (Compose e1 e2)
  (DSL 'Compose (list e1 e2)))

;; 如果 op 是 symbol, 需把 'Rot90 => code=1
(define (trans-name->code nm)
  (define tf (lookup-trans-by-name nm))
  (if tf (TransformationInfo-code tf) #f))

;; 原先的 interp
(define (interp expr obj)
  (match expr
    ;; 组合操作: (DSL 'Compose (list e1 e2))
    [(DSL 'Compose (list e1 e2))
     (define r1 (interp e1 obj))
     (if r1 (interp e2 r1) #f)]

    ;; 单操作, 没子
    [(DSL op #f)
     (define c (trans-name->code op))
     (apply-op c obj)]

    ;; 单操作, 有 sub
    [(DSL op sub)
     (define sub-out (interp sub obj))
     (if sub-out
         (apply-op (trans-name->code op) sub-out)
         #f)]))

;; ---------------------------------------------------------------------
;; 3) 条件化 DSL: If cond => subT else subF, or Base
;; ---------------------------------------------------------------------
;; 3.1 条件结构:
(struct Cond (prop val) #:transparent)
;; prop = 'diagonal? / 'univalued? / ...; val = #t / #f / ...
;; 未来可扩展 bounding-box-size / color / etc.

;; 3.2 条件化 DSL 结构:
;;   'Base => (DSLCond 'Base code #f #f #f)
;;   'If   => (DSLCond 'If #f cond subT subF)
;; 此处用五元组，第一字段存tag, 第二存op, 后面三个当 cond/subT/subF
(struct DSLCond (tag op cond subT subF) #:transparent)

;; 3.3 条件解释器
(define (interp-condition cond obj-info)
  (match-define (Cond prop val) cond)
  (match prop
    ['diagonal?  (equal? (ObjectInfo-diagonal? obj-info) val)]
    ['univalued? (equal? (ObjectInfo-univalued? obj-info) val)]
    [_ #f]))

;; 解释 DSLCond
(define (interp-DSLCond dsl obj-info)
  (match dsl
    ;; Base => single transform code
    [(DSLCond 'Base code #f #f #f)
     (apply-op code (ObjectInfo-obj obj-info))]

    ;; If => if cond => subT else => subF
    [(DSLCond 'If #f cond subT subF)
     (if (interp-condition cond obj-info)
         (interp-DSLCond subT obj-info)
         (interp-DSLCond subF obj-info))]
    [_ #f]))

;; ---------------------------------------------------------------------
;; 4) 统计分析: 在 ParamMatchRecord 层面统计 (diagonal? => transform-code)
;; ---------------------------------------------------------------------
(struct ObjectMatchRecord (in-obj out-obj transform-code details) #:transparent)
(struct ParamMatchRecord (param object-matches) #:transparent)
(struct PairMatchRecord (input-grid output-grid param-match-records) #:transparent)

(define (analyze-param-match-record pmr)
  ;; pmr: (ParamMatchRecord param object-matches)
  ;; 返回一个 hash: key=(list diag? code), val=出现次数
  (define omrs (ParamMatchRecord-object-matches pmr))
  (define diag-count (make-hash))

  (for ([omr (in-list omrs)])
    (define in-obj-info (ObjectMatchRecord-in-obj omr)) ;; 这里 in-obj-info = (ObjectInfo ...)
    (define diag?       (ObjectInfo-diagonal? in-obj-info))
    (define tcode       (ObjectMatchRecord-transform-code omr))
    (hash-update! diag-count
                  (list diag? tcode)
                  (λ (old) (add1 old))
                  0))
  diag-count)

(define (analyze-pair-match-record pmRec)
  (define pmrs (PairMatchRecord-param-match-records pmRec))
  (for/fold ([acc (make-hash)]) ([p (in-list pmrs)])
    (define local-hash (analyze-param-match-record p))
    ;; 合并 local-hash 到 acc
    (for ([k (in-hash-keys local-hash)])
      (define val (hash-ref local-hash k))
      (hash-update! acc k (λ (old) (+ old val)) 0))
    acc))


;; <<<<<<< NEW CODE FOR BOUNDED ROSETTE SEARCH >>>>>>>>>>>>>>>>>>>>>>>
;; 以下插入我们在 2) 中讨论的“有界搜索”示例函数

;; （1）我们用 Rosette 符号化“单层 if-rule”:
;;     prop ∈ props-list, val ∈ boolean, op1/op2 ∈ ops-list
;;     并对所有 (in-obj, out-obj) 约束 (interp-DSLCond T in-obj) = out-obj

;; 为了枚举/限制, 我们可以先把 props-list, ops-list 写死在函数里，
;; 或让调用者传进来都可以。这里演示让调用者传:

(define (search-if-rule-symbolic input-obj-list output-obj-list props-list ops-list)
  ;; 1) 定义符号变量
  (define-symbolic prop-idx integer?)
  (define-symbolic val-choice boolean?)
  (define-symbolic op1-code integer?)
  (define-symbolic op2-code integer?)

  ;; 2) 对 prop-idx 做边界限制: ∈ [0, length(props-list)-1]
  (assert (>= prop-idx 0))
  (assert (< prop-idx (length props-list)))

  ;; 对 op1-code, op2-code 做限制: 必须在 ops-list 中
  ;; Rosette里可以写成： (assert (or (= op1-code (car ops-list)) (= op1-code (cadr ops-list)) ...))
  ;; 这里为了简化，手动展开:
  (define (in-ops-list? x)
    (for/or ([o (in-list ops-list)])
      (= x o)))
  (assert (in-ops-list? op1-code))
  (assert (in-ops-list? op2-code))

  ;; 3) 构造 DSLCond: (If cond => Base op1 : Base op2)
  (define (the-rule)
    (DSLCond 'If
             #f
             (Cond (list-ref props-list prop-idx) val-choice)
             (DSLCond 'Base op1-code #f #f #f)
             (DSLCond 'Base op2-code #f #f #f)))

  ;; 4) 为每个对象对加约束: (interp-DSLCond T in-obj) = out-obj
  (for ([i (in-range (length input-obj-list))])
    (define in-info  (list-ref input-obj-list i))
    (define out-obj (list-ref output-obj-list i))
    (assert (equal? (interp-DSLCond (the-rule) in-info)
                    out-obj)))

  ;; 5) solve:
  (define result (solve))
  (cond
    [(sat? result)
     (define m (model result))
     (define p-idx (lookup m prop-idx))
     (define the-prop (list-ref props-list p-idx))
     (define the-val  (lookup m val-choice))
     (define code1    (lookup m op1-code))
     (define code2    (lookup m op2-code))
     ;; 构造最终 DSLCond
     (DSLCond 'If
              #f
              (Cond the-prop the-val)
              (DSLCond 'Base code1 #f #f #f)
              (DSLCond 'Base code2 #f #f #f))]
    [(unsat? result) #f]
    [else #f]))

;; ========================================================
;; 4) 合成逻辑
;; ========================================================
(define (simple-check in-obj out-obj)
  (for/or ([tf (in-list transformations)])
    ;;; (displayln "simple-check")
    ;;; (displayln (TransformationInfo-name tf))
    ;;; (sleep 1)
    (define cfn (TransformationInfo-check-fn tf))
    (when (and cfn (cfn in-obj out-obj))
      (TransformationInfo-name tf))))

(define-symbolic e integer?)

(define (translate e)
  (define found (lookup-trans-by-code e))
  (if found
      ((TransformationInfo-dsl-maker found) (NoOp))  ; default sub => NoOp
      (error "unrecognized transformation code" e)))

(define (synthesize-transformation input-obj output-obj)
  (define name-result (simple-check input-obj output-obj))
  (cond
    [(symbol? name-result)
     (define tf (lookup-trans-by-name name-result))
     (define c (TransformationInfo-code tf))
     (displayln (format "#hash((e . ~a))" c))
     c ]
    [else
     (define all-conditions
       (and (>= e 0)
            (< e (length transformations))
            (equal? (interp (translate e) input-obj) output-obj)))
     (define result (solve (assert all-conditions)))
     (cond
       [(sat? result)
        (displayln "inoutobj found by SMT!")
        (displayln input-obj)
        (displayln (model result))
        #t]
       [(unsat? result) #f]
       [else (displayln "SMT result: unknown...") #f])]))

;; ---------------------------------------------------------------------
;; 5) 用统计结果启发式构造 if-then DSLCond
;;    假设只关心 diagonal? => code
;; ---------------------------------------------------------------------
;; 从统计结果 (hash (list diag? code) => count)，找出 #t时出现最多的 code & #f时最多的 code
(define (argmax pred lst)
  (if (null? lst)
      #f
      (let ([best (car lst)])
        (for/fold ([acc best]) ([x (in-list (cdr lst))])
          (if (pred x acc) x acc)))))

(define (build-if-rule-based-on-stats diag-hash)
  ;; 找 #t 下计数最大的 transform-code
  (define diag-true-code
    (let ([pairs (for/list ([k (in-hash-keys diag-hash)])
                   (match k
                     [(list #t tcode)
                      (values tcode (hash-ref diag-hash k))]
                     [_ (values #f 0)]))])
      (car (argmax (lambda (a b) (> (cdr a) (cdr b))) pairs))))

  ;; 找 #f 下计数最大的 transform-code
  (define diag-false-code
    (let ([pairs (for/list ([k (in-hash-keys diag-hash)])
                   (match k
                     [(list #f tcode)
                      (values tcode (hash-ref diag-hash k))]
                     [_ (values #f 0)]))])
      (car (argmax (lambda (a b) (> (cdr a) (cdr b))) pairs))))

  (DSLCond 'If
           #f
           (Cond 'diagonal? #t)
           (DSLCond 'Base (or diag-true-code 0) #f #f #f)
           (DSLCond 'Base (or diag-false-code 0) #f #f #f)))

;; ---------------------------------------------------------------------
;; 6) “后处理”阶段：对 pair-match-records 分析 & 构造 if-rule & 测试
;; ---------------------------------------------------------------------
;;; (define pair-match-records '())

(define (post-process-rules!)
  (for ([pmr (in-list pair-match-records)])
    ;; 先做统计:
    (begin
      (define diag-hash (analyze-pair-match-record pmr))
      (define candidate-rule (build-if-rule-based-on-stats diag-hash))
      (displayln (format "Generated if-rule => ~s" candidate-rule))
      (let ([success?
            (for/and ([param-rec (in-list (PairMatchRecord-param-match-records pmr))])
              (define omrs (ParamMatchRecord-object-matches param-rec))
              (for/and ([omr (in-list omrs)])
                (define in-obj-info (ObjectMatchRecord-in-obj omr))
                (define out-obj     (ObjectMatchRecord-out-obj omr))
                (equal? (interp-DSLCond candidate-rule in-obj-info) out-obj)))])

        (displayln (format "Check if-rule success? ~a" success?)))
      ;; 让 begin 的最后是一个表达式:
      'done)

    ;; <<<<<<< NEW USAGE of search-if-rule-symbolic >>>>>>>>>>>>>>>>
    ;; 假设我们这里也演示 Rosette 符号搜索:
    ;; 注意: 需要你先把 pmr里对应的 (in-obj, out-obj) 收集成 input-obj-list / output-obj-list
    ;;       这里仅示例演示, 你要在实际中构建这两个列表

    (let ()
      ;; 仅演示：props-list 先写死 '(diagonal? univalued?)
      ;; ops-list 先写死 '(0 1 2 3 4 5) => 允许NoOp, Rot90, etc.
      ;; 真实项目里，你可按统计结果选 top-3 code => e.g. '(0 1 2)
      (define props-list '(diagonal? univalued?))
      (define ops-list '(0 1 2 3 4 5))

      ;; 构造 input-obj-list / output-obj-list
      ;; （在实际中：遍历 param-match-records, object-match-list ）
      (define input-obj-list '())
      (define output-obj-list '())
      ;; 这里省略: 你可 for-each omr => push (ObjectMatchRecord-in-obj omr) & (ObjectMatchRecord-out-obj omr)

      (define rosette-rule (search-if-rule-symbolic
                            input-obj-list
                            output-obj-list
                            props-list
                            ops-list))
      (when rosette-rule
        (displayln (format "Rosette found if-rule => ~s" rosette-rule))))
    ;; <<<<<<< NEW USAGE end >>>>>>>>>>>>>>>>>>>>>>>>

    ))


;; ---------------------------------------------------------------------
;; 7) process-single-file / main (示例)
;;    这里展示简单框架，保留你原先逻辑
;; ---------------------------------------------------------------------
;; 下面两个 struct, 只示意保留:

;; 全局收集
(define pair-match-records '())

(define (process-single-file json-data)
  ;; 1) 读取训练数据
  (define train-data (hash-ref json-data 'train))

  ;; 2) 用一个 let 包裹外层 for/fold，捕获其多值结果以便打印
  (define-values (all-succeeded? collected-pairs)
    (let-values ([(res-succeeded? res-pairs)
           (for/fold ([acc-succeeded? #t]    ;; 到目前为止是否全部成功
                      [acc-pairs      '()])   ;; 收集的 PairMatchRecord
                     ([pair (in-list train-data)])
             ;; ---------------------------------------
             ;;   针对单个 pair 的处理
             ;; ---------------------------------------
             (define input-grid  (Grid (hash-ref pair 'input)))
             (define output-grid (Grid (hash-ref pair 'output)))

             ;; 提取 input-obj
             (define input-obj-set0 (all-objects-from-grid input-grid))
             (define input-obj-set  (all-objects-00-c0-from-objs input-obj-set0))

             ;; 内层 for/fold: 收集所有能匹配成功的 param => param-records
             (define param-records
               (let ([local-param-records
                      (for/fold ([acc-params '()])
                                ([out-param (in-list param-combinations)])
                        ;; 取出 output objs
                        (define out-obj-set0 (objects-with-params output-grid out-param))
                        (define out-obj-set  (all-objects-00-c0-from-objs out-obj-set0))

                        (define object-match-list '())
                        ;; param 下: “所有 out-obj 必须可解” => for/and
                        (define param-success?
                          (for/and ([out-obj (in-set out-obj-set)])
                            ;; 只要有一个 in-obj 能成功 => for/or
                            (for/or ([in-obj (in-set input-obj-set)])
                              (let ([ok? (synthesize-transformation
                                          (ObjectInfo-obj in-obj)
                                          (ObjectInfo-obj out-obj))])
                                (when ok?
                                  (set! object-match-list
                                        (cons (ObjectMatchRecord in-obj out-obj ok? '())
                                              object-match-list)))
                                ok?))))
                        ;; 如果 param-success? => 新增一个 ParamMatchRecord
                        (if param-success?
                            (cons (ParamMatchRecord out-param object-match-list)
                                  acc-params)
                            acc-params))
                            ])
                 ;; ★ 在内层 for/fold 结束后输出调试日志
                 (displayln (format "[DEBUG] Done param-combinations for this pair. param-records => ~s"
                                    local-param-records))
                 local-param-records)
              )

             ;; 判断该 pair 是否成功
             (define this-pair-success? (not (null? param-records)))

             ;; 构造外层新的累积状态
             (define new-succeeded? (and acc-succeeded? this-pair-success?))
             (define new-pairs
               (if this-pair-success?
                   (cons (PairMatchRecord input-grid output-grid param-records)
                         acc-pairs)
                   acc-pairs))

             ;; ★ 在外层 for/fold 这一轮迭代结束前输出调试日志
             (displayln (format "[DEBUG] after handling ONE pair => success?=~a, total-collected-pairs=~a"
                                this-pair-success?
                                (length new-pairs)))

             (values new-succeeded? new-pairs))])  ;; 结束 for/fold

      ;; ★ for/fold 全部结束后再打印一次整体结果
      (displayln (format "[DEBUG] all pairs processed => all-succeeded?=~a, total=~a"
                         res-succeeded?
                         (length res-pairs)))
      (values res-succeeded? res-pairs)))

  ;; 3) 把本文件处理的 PairMatchRecord 累加到全局
  (set! pair-match-records (append collected-pairs pair-match-records))

  ;; ★ 显示一下最终的 pair-match-records
  (displayln (format "[DEBUG] appended => pair-match-records total=~a"
                     (length pair-match-records)))

  ;; 4) 返回是否全部成功
  all-succeeded?)

;; 可进一步封装一个 process-single-file-logging 或 main 函数
;; 这里仅演示如何在关键处理点输出调试信息


;; 测试：
;; (process-single-file some-json-data)
;; => 若所有 pair 都成功匹配，则返回 #t，否则返回 #f
;; 同时所有 PairMatchRecord 已经被放进全局 pair-match-records 里了。



(define (process-single-file-logging json-data)
  (define fn (hash-ref json-data 'filename))
  (displayln (format " [ ] START => ~a" fn))
  (define success? (process-single-file json-data))
  (if success?
      (displayln (format " [ ] SUCCESS => ~a" fn))
      (displayln (format "[] FAIL    => ~a" fn)))
  success?)

(define (main dir)
  (define all-json (read-all-json-files dir))
  (define total-success
    (for/sum ([json-data (in-list all-json)])
      (if (process-single-file-logging json-data)
          1
          0)))

  (post-process-rules!)

  (displayln (format "[] total-successful-files = ~a" total-success)))

(provide main)

(module+ main
  (command-line
    #:args (dir)
    "Usage: racket your-file.rkt <dir>"
    (main dir)))
