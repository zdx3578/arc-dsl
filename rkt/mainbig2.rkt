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

;; ========================================================
;; 4) 合成逻辑
;; ========================================================
;;; (define (simple-check in-obj out-obj)
;;;   (for/or ([tf (in-list transformations)])
;;;     ;;; (displayln "simple-check")
;;;     ;;; (displayln (TransformationInfo-name tf))
;;;     ;;; (sleep 1)
;;;     (define cfn (TransformationInfo-check-fn tf))
;;;     (when (and cfn (cfn in-obj out-obj))
;;;       (TransformationInfo-name tf))))
(define (simple-check in-obj out-obj)
  (define successful-transformations  ; 在外部定义累积列表
    (for/fold ([result '()  ])               ; 初始值是空列表
              ([tf (in-list transformations)])
      (define cfn (TransformationInfo-check-fn tf))
      (if (and cfn (cfn in-obj out-obj))
          (cons (TransformationInfo-name tf) result)  ; 如果转换成功，添加到列表前面
          result)))  ; 如果转换失败，则继续原列表

  (begin
    ;;; (displayln (format "All--------------------- successful transformations: ~a" successful-transformations))
    successful-transformations))  ; 返回所有成功的转换记录


(define-symbolic e integer?)

(define (translate e)
  (define found (lookup-trans-by-code e))
  (if found
      ((TransformationInfo-dsl-maker found) (NoOp))  ; default sub => NoOp
      (error "unrecognized transformation code" e)))

(define (synthesize-transformation input-obj output-obj)
  (define name-result (simple-check input-obj output-obj))
  (cond
    [(not (empty? name-result))
    ;;;  (define tf (lookup-trans-by-name name-result))
    ;;;  (define c (TransformationInfo-code tf))
     (displayln (format "#hash((e . ~a))" name-result))
     name-result ]
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
  ;; pred: (lambda (x y) (boolean?)) 判断 x 是否比 y “更好”
  ;; lst : list of elements (each element is a pair/list)
  (cond
    [(null? lst) #f]
    [else
     (for/fold ([best (car lst)])
               ([x (in-list (cdr lst))])
       (if (pred x best) x best))]))

(define (build-if-rule-based-on-stats diag-hash)
  ;; diag-hash: (Hash (list diag? code) => count)
  ;; 1) 找 (#t, code) 出现次数最多的 code
  (define diag-true-code
    (let ([pairs
           (for/list ([k (in-hash-keys diag-hash)])
             (match k
               [(list #t tcode)
                (list tcode (hash-ref diag-hash k))]
               [_
                (list #f 0)]))])   ;; 返回 '(tcode count) or '(#f 0)

      (define best (argmax (lambda (a b) (> (cadr a) (cadr b))) pairs))
      (if best (car best) 0)))  ;; 取 best 的第一个元素就是 tcode，第二个是 count

  ;; 2) 找 (#f, code) 出现次数最多的 code
  (define diag-false-code
    (let ([pairs
           (for/list ([k (in-hash-keys diag-hash)])
             (match k
               [(list #f tcode)
                (list tcode (hash-ref diag-hash k))]
               [_
                (list #f 0)]))])

      (define best (argmax (lambda (a b) (> (cadr a) (cadr b))) pairs))
      (if best (car best) 0)))

  ;; 3) 构造一个简单的单层 if-rule: if diagonal? => diag-true-code else diag-false-code
  (DSLCond 'If
           #f
           (Cond 'diagonal? #t)
           (DSLCond 'Base diag-true-code #f #f #f)
           (DSLCond 'Base diag-false-code #f #f #f)))

;; ---------------------------------------------------------------------
;; 6) “后处理”阶段：对 pair-match-records 分析 & 构造 if-rule & 测试
;; ---------------------------------------------------------------------
;; 假设仅做一个简单的统计 => 生成一个 candidate-rule
;; 再验证 candidate-rule 在 pair-match-records 里是否都能成功
(define (post-process-rules! pmrs)
  (define diag-hash (make-hash))  ;; 用来全局计数
  ;; 先把 pmrs 里所有的对象匹配情况汇总
  (for ([pmr (in-list pmrs)])
    (define local-hash (analyze-pair-match-record pmr)) ; 这是你之前的函数
    ;; 合并到 diag-hash
    (for ([k (in-hash-keys local-hash)])
      (define val (hash-ref local-hash k))
      (hash-update! diag-hash k (λ (old) (+ old val)) 0)))

  ;; 构造 if-rule
  (define candidate-rule (build-if-rule-based-on-stats diag-hash)) ;; 也是你之前的函数
  (displayln candidate-rule)

  ;; 逐条验证
  (define success?
    (for/and ([pmr (in-list pmrs)])
      (for/and ([param-rec (in-list (PairMatchRecord-param-match-records pmr))])
        (define omrs (ParamMatchRecord-object-matches param-rec))
        (for/and ([omr (in-list omrs)])
          (define in-obj-info (ObjectMatchRecord-in-obj omr))
          (define out-obj     (ObjectMatchRecord-out-obj omr))
          (equal? (interp-DSLCond candidate-rule in-obj-info) out-obj)))))

  (displayln (format "Generated if-rule => ~s" candidate-rule))
  (displayln (format "Check if-rule success? ~a" success?))

  candidate-rule)

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
             (define input-obj-set (all-objects-from-grid input-grid))
            ;;;  (define input-obj-set  (all-objects-00-c0-from-objs input-obj-set0))

             ;; 内层 for/fold: 收集所有能匹配成功的 param => param-records
             (define param-records
               (let ([local-param-records
                      (for/fold ([acc-params '()])
                                ([out-param (in-list param-combinations)])
                        ;; 取出 output objs
                        (define out-obj-set (objects-with-params output-grid out-param))
                        ;;; (define out-obj-set  (all-objects-00-c0-from-objs out-obj-set0))

                        (define object-match-list '())
                        ;; param 下: “所有 out-obj 必须可解” => for/and
                        (define param-success?
                          (for/and ([out-obj (in-set out-obj-set)])
                            ;; 第 1 步：遍历 input-obj-set，做 regular 的匹配
                            (let ([found-regular?
                                  (for/or ([in-obj (in-set input-obj-set)])
                                    (let ([code-regular
                                            (synthesize-transformation
                                              (ObjectInfo-obj in-obj)
                                              (ObjectInfo-obj out-obj))])
                                      (when code-regular
                                        (set! object-match-list
                                              (cons (ObjectMatchRecord in-obj out-obj code-regular '())
                                                    object-match-list)))
                                      code-regular))])
                              ;; 第 2 步：如果上面那一步 found-regular? 为 #f，就再尝试 shift 匹配
                              (or found-regular?
                                  (for/or ([in-obj (in-set input-obj-set)])
                                    (let ([code-shift
                                          (synthesize-transformation
                                            (ObjectInfo-obj (shift-obj-to-0-0-0 in-obj))
                                            (ObjectInfo-obj (shift-obj-to-0-0-0 out-obj)))])
                                      (when code-shift
                                        (set! object-match-list
                                              (cons (ObjectMatchRecord in-obj out-obj code-shift '())
                                                    object-match-list)))
                                      code-shift))))
                                )                                )
                        ;; 如果 param-success? => 新增一个 ParamMatchRecord
                        (if param-success?
                            (cons (ParamMatchRecord out-param object-match-list)
                                  acc-params)
                            acc-params))
                            ])
                 ;; ★ 在内层 for/fold 结束后输出调试日志
                 (displayln (format "[DEBUG] Done param-combinations for this pair. param-records => ~s"
                                    local-param-records))
                 local-param-records)              )
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
    ;; 3) 做后处理: 生成 if-rule / 统计
  (define candidate-rule (post-process-rules! pair-match-records))

  ;; 4) 若有 test 数据则验证
  (define test-data (hash-ref json-data 'test #f))
  (define test-success? (if test-data
                           (verify-test-data test-data candidate-rule)  ;; 上面示例
                           #t)) ;; 如果没有 test 就算成功

  (displayln (format "Test-data check => ~a" test-success?))
  ;; 最终只要所有 pair 匹配成功 + 测试成功 => 整体成功
  test-success?
  ;; 4) 返回是否全部成功
  all-succeeded?)

(define (process-single-file-logging json-data)
  (define fn (hash-ref json-data 'filename))
  (displayln (format " [ ] START => ~a" fn))
  (define success? (process-single-file json-data))
  (if success?
      (displayln (format " [ ] SUCCESS => ~a" fn))
      (displayln (format "[] FAIL    => ~a" fn)))
  (set! pair-match-records '())  ;; 清空全局记录   下一个文件处理的时候是空状态
  success?)



;; 假设 candidate-rule 是一个 DSLCond
(define (apply-if-rule in-grid rule)
  ;; 拿到 in-grid 的所有 objs，或者根据需要处理
  ;; 这里只示意：针对每个对象 interpret 后构造一个新的输出网格
  ;; 实际的实现取决于你自己的结构
  #f)  ;; TODO: 你要自行实现


(define (verify-test-data test-data candidate-rule)
  (for/and ([td (in-list test-data)])
    (define in-grid  (Grid (hash-ref td 'input)))
    (define out-grid (Grid (hash-ref td 'output)))    ;; 真值
    (define predicted (apply-if-rule in-grid candidate-rule))
    (equal? predicted out-grid)))  ;; 看你如何定义 equals


(define (main dir)
  (define all-json (read-all-json-files dir))
  (define total-success
    (for/sum ([json-data (in-list all-json)])
      (if (process-single-file-logging json-data)
          1
          0)))

  (displayln (format "[] total-successful-files = ~a" total-success)))

(provide main)

(module+ main
  (command-line
    #:args (dir)
    "Usage: racket your-file.rkt <dir>"
    (main dir)))
