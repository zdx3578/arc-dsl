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
    (lambda (obj)  (rotate90-info obj))
    (lambda (i o) (equal? (rotate90-info i) o))
    (lambda (sub) (Rot90 sub)))

   (TransformationInfo
    'HMirror
    2
    (lambda (obj)  (hmirror-info obj))
    (lambda (i o) (equal? (hmirror-info i) o))
    (lambda (sub) (HMirror sub)))

   (TransformationInfo
    'VMirror
    3
    (lambda (obj)  (vmirror-info obj))
    (lambda (i o) (equal? (vmirror-info i) o))
    (lambda (sub) (VMirror sub)))

   (TransformationInfo
    'CMirror
    4
    (lambda (obj) (cmirror-info obj))
    (lambda (i o) (equal? (cmirror-info i) o))
    (lambda (sub) (CMirror sub)))

   (TransformationInfo
    'DMirror
    5
    (lambda (obj) (dmirror-info obj))
    (lambda (i o) (equal? (dmirror-info i) o))
    (lambda (sub) (DMirror sub)))
   ))

;; 查找 transformation
(define (lookup-trans-by-code c)
  (for/first ([tf (in-list transformations)])
    (if (= c (TransformationInfo-code tf))
        tf
        #f)))

(define (lookup-trans-by-name nm)
  (let ([found (filter (lambda (tf)
                         (eq? nm (TransformationInfo-name tf)))
                       transformations)])
    (if (null? found)
        #f
        (car found))))

;; 统一 apply-op: code => transformations
;; 同时支持 integer/symbol/以及 list-of-symbols
(define (apply-op code-or-codes obj)
  (cond
    ;; 若是整数 => lookup-trans-by-code
    [(integer? code-or-codes)
     (define tf (lookup-trans-by-code code-or-codes))
     (if tf
         ((TransformationInfo-apply-fn tf) obj)
         #f)]

    ;; 若是符号 => lookup-trans-by-name
    [(symbol? code-or-codes)
     (define tf (lookup-trans-by-name code-or-codes))
     (if tf
         ((TransformationInfo-apply-fn tf) obj)
         #f)]

    ;; 若是列表(符号集合) => 依次 apply
    [(and (list? code-or-codes)
          (for/and ([c (in-list code-or-codes)]) (symbol? c)))
     (for/fold ([acc obj])
               ([c (in-list code-or-codes)])
       (define tf (lookup-trans-by-name c))
       (if tf
           ((TransformationInfo-apply-fn tf) acc)
           acc))]

    [else
     (error "apply-op: unexpected code/codes" code-or-codes)]))

;; -----------------------------------------------------------
;; 2) DSL 定义 & interp (含 Compose)
;; -----------------------------------------------------------
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
  (if tf
      (TransformationInfo-code tf)
      #f))

;; DSL 的解释器
(define (interp expr obj)
  (match expr
    ;; 组合操作: (DSL 'Compose (list e1 e2))
    [(DSL 'Compose (list e1 e2))
     (define r1 (interp e1 obj))
     (if r1
         (interp e2 r1)
         #f)]
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
(struct Cond (prop val) #:transparent)
;; prop = 'diagonal? / 'univalued? / ...
;; val = #t/#f/...
;; 未来可扩展 bounding-box-size / color / etc.

;; 条件化 DSL 的结构
;; tag = 'Base / 'If
;; 'Base => (DSLCond 'Base code #f #f #f)
;; 'If   => (DSLCond 'If #f cond subT subF)
(struct DSLCond (tag op cond subT subF) #:transparent)

(define (interp-condition cond obj-info)
  (match-define (Cond prop val) cond)
  (match prop
    ['diagonal?  (equal? (ObjectInfo-ismove000 obj-info) val)]
    ['univalued? (equal? (ObjectInfo-ismove000 obj-info) val)]
    [_ #f])) ;; 需要自行扩展

(define (interp-DSLCond dsl obj-info)
  (match dsl
    ;; Base => single transform code
    [(DSLCond 'Base code #f #f #f)
     (apply-op code obj-info)]
    ;; If => if cond => subT else => subF
    [(DSLCond 'If #f cond subT subF)
     (if (interp-condition cond obj-info)
         (interp-DSLCond subT obj-info)
         (interp-DSLCond subF obj-info))]
    [_ #f]))

;; ---------------------------------------------------------------------
;; 4) 一些简单函数: simple-check / synthesize-transformation 等
;; ---------------------------------------------------------------------
(define (simple-check in-obj out-obj)
  (define successful-transformations
    (for/fold ([result '()]) ([tf (in-list transformations)])
      (define cfn (TransformationInfo-check-fn tf))
      (if (and cfn (cfn in-obj out-obj))
          (cons (TransformationInfo-name tf) result)
          result)))
  successful-transformations)

;; 用 Rosette 符号变量 e 来解某个单一变换
(define-symbolic e integer?)
(define (translate e)
  (define found (lookup-trans-by-code e))
  (if found
      ((TransformationInfo-dsl-maker found) (NoOp))
      (error "unrecognized transformation code" e)))

(define (synthesize-transformation input-obj output-obj)
  (define name-result (simple-check input-obj output-obj))
  (cond
    [(not (empty? name-result))
      (displayln (format "simple-check found! => ~a " name-result))
     ;; 这里 name-result 可能是一个列表(比如'(HMirror)), 也可能多个
     name-result]
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
       [else
        (displayln "SMT result: unknown...")
        #f])]))

;; ---------------------------------------------------------------------
;; 5) 多个 PairMatchRecordEx 的统计分析与后处理
;; ---------------------------------------------------------------------

;; --------------------------------------------
;; 1) 数据结构
;; --------------------------------------------
(struct ObjectMatchRecord
  (in-obj
   out-obj
   transform-code      ;; 列表, e.g. '(HMirror Rotate90 ...)
   details)
  #:transparent)

(struct ParamMatchRecord
  (param               ;; e.g. (#f #t #t)
   object-matches)     ;; (listof ObjectMatchRecord)
  #:transparent)

(struct PairMatchRecord
  (input-grid
   output-grid
   pair-id
   param-match-records)  ;; (listof ParamMatchRecord)
  #:transparent)

;; 扩展结构: PairMatchRecordEx
;; 多了 param-analysis, pair-id，以及可选的 param-rules（用于存储提炼出的子规则）
(struct PairMatchRecordEx
  (input-grid
   output-grid
   param-match-records
   param-analysis      ;; param->(transform->count) 或更多统计
   pair-id
   param-rules)        ;; param->(可能的 DSL 或其他结构)
  #:transparent
  #:constructor-name make-PairMatchRecordEx)

;; --------------------------------------------
;; 2) 对 ParamMatchRecord 做分析 => (transform->count)
;; --------------------------------------------
(define (analyze-single-param-match-record pmr)
  (define result (make-hash))
  (for ([omr (in-list (ParamMatchRecord-object-matches pmr))])
    (for ([code (in-list (ObjectMatchRecord-transform-code omr))])
      (hash-update! result code add1 0)))
  ; (displayln (format "analyze-single-param-match-record => ~s" result))
  result)

(define (analyze-param-match-records pmrs)
  (define param->transform-count (make-hash))
  (for ([pmr (in-list pmrs)])
    (define param-key (ParamMatchRecord-param pmr))
    (define single-hash (analyze-single-param-match-record pmr))
    (define existing-hash
      (hash-ref param->transform-count param-key (lambda () (make-hash))))
    (for ([t (in-hash-keys single-hash)])
      (define val (hash-ref single-hash t))
      (hash-update! existing-hash t (λ (old) (+ old val)) 0))
    (hash-set! param->transform-count param-key existing-hash))
  (displayln (format "analyze-param-match-records => ~s" param->transform-count))
  param->transform-count)

;; --------------------------------------------
;; 3) 合并两个 "param->transform->count" 哈希表
;; --------------------------------------------
(define (merge-two-level-hash! main other)
  (for ([param (in-hash-keys other)])
    (define other-sub (hash-ref other param))
    (define main-sub (hash-ref main param (lambda () (make-hash))))
    (for ([t (in-hash-keys other-sub)])
      (define val (hash-ref other-sub t))
      (hash-update! main-sub t (λ (old) (+ old val)) 0))
    (hash-set! main param main-sub))
    (displayln (format "merge-two-level-hash! => ~s" main))
  main)

;; --------------------------------------------
;; 4) 对多个 PairMatchRecordEx 累加 => 全局 param->transform->count
;; --------------------------------------------
(define (collect-global-param-analysis pair-records-ex)
  (for/fold ([acc (make-hash)])
            ([prex (in-list pair-records-ex)])
    (define local-hash (PairMatchRecordEx-param-analysis prex))
    (merge-two-level-hash! acc local-hash)
    ; (displayln (format "collect-global-param-analysis => ~s" acc))
    acc))

;; --------------------------------------------
;; 新增：从 ObjectMatchRecord 列表里，提炼一个“单一变换”或其他更复杂的模式
;;       用于示例：对所有 (in-obj, out-obj) 是否都有某个共同变换？
;; --------------------------------------------
(define (extract-consistent-rule-for-param obj-match-list)
  ;; obj-match-list: (listof ObjectMatchRecord)
  ;; 简单示例：若所有 ObjectMatchRecord 的 transform-code 列表中都含有同一个变换symbol，则返回它。
  ;; 实际可以更复杂: 用计数统计出现频次, 也可组合 Compose, etc.
  (define all-transform-lists
    (map ObjectMatchRecord-transform-code obj-match-list)) ;; => list of lists
  (define flatten-all (apply append all-transform-lists))  ;; => 汇总所有 code

  (define transform->freq (make-hash))
  (for ([code (in-list flatten-all)])
    (hash-update! transform->freq code add1 0))

  ;; 在这里简单处理：若某变换 code 的出现频数 == (length obj-match-list) 说明每个对象匹配都能用它
  (define consistent-candidates
    (for/list ([k (in-hash-keys transform->freq)]
               #:when (= (hash-ref transform->freq k) (length obj-match-list)))
      k))
  ;; 若找到多个就随便 pick 一个(或用优先级排序), 若没找到就返回 #f
  (cond
    [(null? consistent-candidates) #f]
    [else
     ;; pick the first
     (car consistent-candidates)]))

;; --------------------------------------------
;; 新增：对单个 PairMatchRecordEx 进行更高层次分析
;;       将每个 param 对应的所有 obj-match-list 提炼成一个 DSL (或简单 transform)
;; --------------------------------------------
(define (build-param-rules-from-PairMatchRecordEx pmr-ex)
  (define pmrs (PairMatchRecordEx-param-match-records pmr-ex))

  ;; 目标：返回一个 param->(可能是 symbol 或 DSLCond 'Base transformCode)
  (define param->rule (make-hash))

  (for ([pmr (in-list pmrs)])
    (define param-key (ParamMatchRecord-param pmr))
    (define omr-list (ParamMatchRecord-object-matches pmr))
    (define found-transform (extract-consistent-rule-for-param omr-list))
    ; (displayln (format "build-param-rules fun : found transform => ~s" found-transform))
    (define saved-rule
      (cond
        [(symbol? found-transform)
         ;; 若找到一个 symbol, 就构造 DSLCond 'Base
         (DSLCond 'Base
                  found-transform
                  #f #f #f)]
        [else
        ; (displayln (format "build-param-rules fun : save rule : ------- not found  => ~s" found-transform))
         (make-hash)])) ;; #f 表示没法提炼出单一变换

    (hash-set! param->rule param-key saved-rule))

  param->rule)



(define (post-process-pair-records-ex pair-records-ex)
  (for/list ([prex (in-list pair-records-ex)])
    ;; 生成新的 param-rules
    (define p-rules (build-param-rules-from-PairMatchRecordEx prex))
    (displayln (format "p-rules => ~s" p-rules))
    ;; 返回一个新的 PairMatchRecordEx
    (make-PairMatchRecordEx
     (PairMatchRecordEx-input-grid prex)
     (PairMatchRecordEx-output-grid prex)
     (PairMatchRecordEx-param-match-records prex)
     (PairMatchRecordEx-param-analysis prex)
     (PairMatchRecordEx-pair-id prex)
     p-rules)))


;; --------------------------------------------
;; 将每个 PairMatchRecordEx 里提炼出来的 param-rules, 全局合并成一个大的 if-then DSLCond
;; 简单示例：若 param 可能是 (#f #f #f), (#t #f #f), ... 就做一连串 if-else
;; --------------------------------------------
(define (build-global-param-based-rule pair-records-ex)
  ;; 先收集所有 param
  (define param-set (mutable-set))
  (for ([prex (in-list pair-records-ex)])
    (define pr (PairMatchRecordEx-param-rules prex))
    (displayln (format "build-global-param-based-rule : pr => ~s" pr))
    (for ([p (in-hash-keys pr)])
      (set-add! param-set p)))
  (displayln (format "\nbuild-global-param-based-rule : param-set => ~s" param-set))

  ;; 这里示范：对 param-set 里每个 param 选一个“最常见/最简单”的 rule(或就直接用第一条)
  ;; 实际可以再合并“多文件/pair”的一致性；此处演示：直接 pick 第一个 PairMatchRecordEx 的 param-rules
  (define param->final-rule (make-hash))

  (for ([p (in-set param-set)])
    (define candidate-rules-for-p
      (for/list ([prex (in-list pair-records-ex)])
        (hash-ref (PairMatchRecordEx-param-rules prex) p #f)))
    (displayln (format "build-global-param-based-rule : candidate-rules-for-p => ~s" candidate-rules-for-p))
    ;; candidate-rules-for-p 可能收集到多个 DSLCond / #f
    ;; 简单做法：若有非#f的 rule 就 pick 第一个
    (define chosen
      (let ([non-false (removef #f candidate-rules-for-p)])
        (if (null? non-false)
            #f
            (car non-false))))
    (hash-set! param->final-rule p chosen))
    (displayln (format "build-global-param-based-rule : param->final-rule => ~s" param->final-rule))

  ;; 最后把 param->final-rule 做成一个 if-then DSLCond (或多层 if-then)
  ;; 这里示范只支持二元 param(可能是 #t/#f), 若是多元可做更复杂处理
  ;; 为了演示，我们把 param 当作一个 tuple (#f #t #f), 你可根据需要自己写 condition-check
  ;;
  ;; 简单思路：构造一个“分支链”：
  ;;  if (param = p1) => rule1
  ;;  else if (param = p2) => rule2
  ;;  else => NoOp
  ;;
  ;; 注意: 这里 param 可能不止 2-3 种, 你可以把它做成一个递归拼装
  (define param-list (set->list param-set))

  (define (make-cond-dsl lst)
    (cond
      [(null? lst) (DSLCond 'Base 'NoOp #f #f #f)]
      [else
       (define p (car lst))
       (define sub-rule (hash-ref param->final-rule p #f))
       ;; 构造 condition
       ;; 需要你自己实现 (Cond 'param=?? p)
       ;; 这里仅示意: 我们不再细分 param 各分量了, 只做一个伪 "param= p" 的测试
       (define cond-exp (Cond 'param= p)) ;; 你需要自己实现 interp-condition 里的相关匹配
       (define then-dsl (if sub-rule sub-rule (DSLCond 'Base 'NoOp #f #f #f)))
       (define else-dsl (make-cond-dsl (cdr lst)))
       (DSLCond 'If
                #f
                cond-exp
                then-dsl
                else-dsl)]))

  (make-cond-dsl param-list))

;; --------------------------------------------
;; 6) “示例”后处理: post-process-rules!
;; --------------------------------------------
(define (post-process-rules! pmrs)
  (displayln (format "Total pair-match-records => ~a" (length pmrs)))

  ;; 将 PairMatchRecord -> PairMatchRecordEx(含 param-analysis)
  ;; 并搜集到 pair-records-ex
  (define pair-records-ex
    (for/list ([p (in-list pmrs)]
               [idx (in-naturals)])
      (define input-grid (PairMatchRecord-input-grid p))
      (define output-grid (PairMatchRecord-output-grid p))
      (define param-recs (PairMatchRecord-param-match-records p))
      (define param-analysis (analyze-param-match-records param-recs))
      (make-PairMatchRecordEx
       input-grid
       output-grid
       param-recs
       param-analysis
       (PairMatchRecord-pair-id p)
       (make-hash)))) ;; 先令 param-rules=#f
  ; (displayln (format "\nTotal pair-records-ex => ~a content ~s" (length pair-records-ex) pair-records-ex))
  ;; 进一步处理：对每个 PairMatchRecordEx, 补充 param-rules
  (define pair-records-ex-updated
    (post-process-pair-records-ex pair-records-ex))

    ; (displayln (format "\nTotal pair-records-ex-updated => ~a content ~s" (length pair-records-ex-updated) pair-records-ex-updated))


  ;; 然后可以尝试整合所有 pair => 生成一个全局大一统的 param-based 规则
  (define final-rule
    (build-global-param-based-rule pair-records-ex-updated))

  (displayln (format "Generated final-rule => ~s" final-rule))

  ;; 这里可以再验证 final-rule 的适用度
  (define success?
    (for/and ([prex (in-list pair-records-ex-updated)])
      (for/and ([pmr (in-list (PairMatchRecordEx-param-match-records prex))])
        (define omrs (ParamMatchRecord-object-matches pmr))
        (for/and ([omr (in-list omrs)])
          (define in-obj-info (ObjectMatchRecord-in-obj omr))
          (define out-obj (ObjectMatchRecord-out-obj omr))
          (equal? (interp-DSLCond final-rule in-obj-info)
                  out-obj)))))
  (displayln (format "Check final-rule success? => ~a" success?))

  ;; 返回最终生成的规则
  final-rule)


;; ---------------------------------------------------------------------
;; 一个演示性的 process-single-file 函数
;; 说明如何在得到 PairMatchRecordEx 后做后处理并生成最终规则
;; ---------------------------------------------------------------------
(define pair-match-records '()) ;; “全局”收集

(define (process-single-file json-data)
  (define train-data (hash-ref json-data 'train))
  (define-values (all-succeeded? collected-pairs-ex)
    (let ([outer-iter 0])
    (let-values ([(res-succeeded? res-pairs-ex)
                  (for/fold ([acc-succeeded? #t]
                             [acc-pairs-ex '()]
                             )
                            ([pair (in-list train-data)]  )
                    (define input-grid (Grid (hash-ref pair 'input)))
                    (define output-grid (Grid (hash-ref pair 'output)))
                    (define raw-id (hash-ref pair 'id #f))
                    (define the-pair-id
                      (if raw-id
                          raw-id
                          (format "auto-pair-~a" outer-iter  )))
                    (set! outer-iter (add1 outer-iter))

                    ;; 提取 input-obj
                    (define input-obj-set (all-objects-from-grid input-grid))

                    ;; 这里仅示意: 你自己定义 param-combinations / objects-with-params
                    (define param-match-records
                      (let ([local-param-records
                             (for/fold ([acc-params '()])
                                       ([out-param (in-list param-combinations)])
                               (define out-obj-set (objects-with-params output-grid out-param))
                               (define object-match-list '())
                               (define param-success?
                                 (for/and ([out-obj (in-set out-obj-set)])
                                   (let ([found-regular?
                                          (for/or ([in-obj (in-set input-obj-set)])
                                            (let ([code-regular
                                                   (synthesize-transformation
                                                    in-obj out-obj)])
                                              (when code-regular
                                                (set! object-match-list
                                                      (cons
                                                       (ObjectMatchRecord
                                                        (smallnoobj-objinfo-obj in-obj)
                                                        (smallnoobj-objinfo-obj out-obj)
                                                        ; "in obj"  "out obj"
                                                        code-regular
                                                        '("-----0-----" #f))
                                                       object-match-list)))
                                              code-regular))])
                                     (or found-regular?
                                         (for/or ([in-obj (in-set input-obj-set)])
                                           (let ([code-shift
                                                  (synthesize-transformation
                                                   (shift-obj-to-0-0-0 in-obj)
                                                   (shift-obj-to-0-0-0 out-obj))])
                                             (when code-shift
                                               (set! object-match-list
                                                     (cons
                                                      (ObjectMatchRecord
                                                       (smallnoobj-objinfo-obj in-obj)
                                                       (smallnoobj-objinfo-obj out-obj)
                                                      ; "in obj"  "out obj"
                                                       code-shift
                                                       '("-----0-----" #t))
                                                      object-match-list)))
                                             code-shift))))))
                               (if param-success?
                                   (cons (ParamMatchRecord out-param object-match-list)
                                         acc-params)
                                   acc-params))])
                          (displayln
                            (format "[DEBUG] Done param-combinations for this pair. param-records => ~s"
                                    local-param-records))
                        local-param-records))

                    (define this-pair-success? (not (null? param-match-records)))
                    (define new-succeeded? (and acc-succeeded? this-pair-success?))

                    (define new-pair (PairMatchRecord input-grid output-grid the-pair-id param-match-records))
                    (define new-acc-pairs (if this-pair-success?
                                              (cons new-pair acc-pairs-ex)
                                              acc-pairs-ex))
                    (values new-succeeded? new-acc-pairs ))])
      (values res-succeeded? res-pairs-ex))))

  ;; 把本文件处理结果追加到一个全局 pair-match-records 中
  (set! pair-match-records (append collected-pairs-ex pair-match-records))
  ; (displayln (format "\n\n Total pair-match-records lenght ~a  content: => ~a" (length pair-match-records)  pair-match-records ))
  (displayln (format "\n\n Total pair-match-records lenght ~a  content: => " (length pair-match-records)   ))

  ;; 这里再计算 globalParamAnalysis
  (define globalParamAnalysis
    (collect-global-param-analysis
     (for/list ([pmr (in-list pair-match-records)]
                [idx (in-naturals)])
       (define input-grid (PairMatchRecord-input-grid pmr))
       (define output-grid (PairMatchRecord-output-grid pmr))
       (define param-recs (PairMatchRecord-param-match-records pmr))
       (define param-analysis (analyze-param-match-records param-recs))
       (make-PairMatchRecordEx
        input-grid
        output-grid
        param-recs
        param-analysis
        (PairMatchRecord-pair-id pmr)
        (make-hash)))))

  (displayln (format "\n [DEBUG] globalParamAnalysis => ~s" globalParamAnalysis))

  ;; 做后处理: 生成 “统一规则”
  (define candidate-rule (post-process-rules! pair-match-records))

  ;; 若有 test 数据则验证
  (define test-data (hash-ref json-data 'test #f))
  (define test-success?
    (if test-data
        (verify-test-data test-data candidate-rule)
        #t))

  (displayln (format "Test-data check => ~a" test-success?))



  ;; 返回 (all-succeeded? globalParamAnalysis) 仅作演示
  (values all-succeeded? globalParamAnalysis))

;; 一个简单的包装: process-single-file-logging
(define (process-single-file-logging json-data)
  (define fn (hash-ref json-data 'filename))
  (displayln (format " [ ] START => ~a" fn))
  (define-values (ok? gpa) (process-single-file json-data))
  (if ok?
      (displayln (format " [ ] SUCCESS => ~a" fn))
      (displayln (format " [ ] FAIL    => ~a" fn)))
  ;; 处理完后清空全局 pair-match-records，避免下个文件相互干扰
  (set! pair-match-records '())
  ok?)

;; 假设 candidate-rule 是一个 DSLCond
(define (apply-if-rule in-grid rule)
  ;; 根据 rule, 对 in-grid 进行变换, 这里只是演示，返回 #f
  ;; 实际要根据 param 的判断(或直接对所有 objects 做 interp-DSLCond)构造 output-grid
  #f)

(define (verify-test-data test-data candidate-rule)
  (for/and ([td (in-list test-data)])
    (define in-grid (Grid (hash-ref td 'input)))
    (define out-grid (Grid (hash-ref td 'output))) ;; 期望值
    (define predicted (apply-if-rule in-grid candidate-rule))
    (equal? predicted out-grid)))

(define (main dir)
  (define all-json (read-all-json-files dir))
  (define total-success
    (for/sum ([json-data (in-list all-json)])
      (if (process-single-file-logging json-data)
          1
          0)))
  (displayln (format "[] total-successful-files = ~a" total-success)))

(provide main)

(define dir "/Users/zhangdexiang/github/VSAHDC/arc-dsl/rkt/data")
; (define dir "/Users/zhangdexiang/github/VSAHDC/arc-dsl/rkt/data")

(module+ main
  ; (command-line
  ;  #:args (dir)
  ;  "Usage: racket your-file.rkt <dir>"
   (main dir))
