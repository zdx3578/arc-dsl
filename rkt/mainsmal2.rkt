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
    (lambda (obj)  (rotate90 obj))
    (lambda (i o) (equal? (rotate90 i) o))
    (lambda (sub) (Rot90 sub)))

   (TransformationInfo
    'HMirror
    2
    (lambda (obj)  (hmirror obj))
    (lambda (i o) (equal? (hmirror i) o))
    (lambda (sub) (HMirror sub)))

   (TransformationInfo
    'VMirror
    3
    (lambda (obj)  (vmirror obj))
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
    ['diagonal?  (equal? (ObjInf-obj obj-info) val)]
    ['univalued? (equal? (ObjInf-obj obj-info) val)]
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
      ; (displayln (format "simple-check found! => ~a " name-result))
     ;; 这里 name-result 可能是一个列表(比如'(HMirror)), 也可能多个
     (reverse name-result)]
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
(struct ObjMR
  (in-obj
   out-obj
   transform-code      ;; 列表, e.g. '(HMirror Rotate90 ...)
   details)
  #:transparent)

(struct ParamMatchRecord
  (param               ;; e.g. (#f #t #t)
   object-matches)     ;; (listof ObjMR)
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


; (define manager (new id-manager%))
;; ---------------------------------------------------------------------
;; 一个演示性的 process-single-file 函数
;; 说明如何在得到 PairMatchRecordEx 后做后处理并生成最终规则
;; ---------------------------------------------------------------------
(define pair-match-records '()) ;; “全局”收集

(define (process-single-file json-data)
  (define object-match-listall '())
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
                          (format "pair-~a" outer-iter  )))
                    (set! outer-iter (add1 outer-iter))

                    ;; 提取 input-obj
                    ; (define input-obj-set (all-objects-from-grid-with-ids input-grid the-pair-id))
                    (define input-obj-set (all-objects-from-grid the-pair-id 'input input-grid))
                    (define input-obj-set000  (all-objects-00-c0-from-objs input-obj-set))

                    ;; 这里仅示意: 你自己定义 param-combinations / objects-with-params
                    (define param-match-records
                      (let ([local-param-records
                             (for/fold ([acc-params '()])
                                       ([out-param (in-list param-combinations)])
                               (define out-obj-set (objects-with-params the-pair-id 'out output-grid out-param))
                                (define out-obj-set000  (all-objects-00-c0-from-objs out-obj-set))
                               (define object-match-list '())
                               (define param-success?
                                ;  (for/and ([out-obj (in-set out-obj-set)])
                                (for/and ([out-obj (in-set out-obj-set000)])
                                   (let ([found-regular?
                                          (for/or ([in-obj (in-set input-obj-set000)])
                                          ; (for/or ([in-obj (in-set input-obj-set000)])
                                            (let ([code-regular
                                                   (synthesize-transformation
                                                    (ObjInf-obj-00 in-obj)
                                                    (ObjInf-obj-00 out-obj)
                                                    )])
                                                  ; (displayln (format "[-~s-------------------------------DEBUG] synthesize-transformation =>  ~s ~s"
                                                  ;           code-regular in-obj out-obj))
                                              (when code-regular
                                                (let ([new-record (ObjMR
                                                              (smallnoobj-objinfo-obj in-obj)
                                                              (smallnoobj-objinfo-obj out-obj)
                                                              code-regular
                                                              '("00"))])
                                                  (set! object-match-list (cons new-record object-match-list))
                                                  (set! object-match-listall (cons new-record object-match-listall)))
                                              )
                                              code-regular))])
                                     (or found-regular?
                                        ; (define out-obj000 (shift-obj-to-0-0-0 out-obj))
                                         (for/or ([in-obj000 (in-set input-obj-set000)])
                                        ;  (displayln (format " [DEBUG] in-obj  shift-obj-to-0-0-0 => " ))
                                           (let ([code-shift
                                                  (synthesize-transformation
                                                  (ObjInf-obj-000 in-obj000)
                                                  (ObjInf-obj-000 (shift-obj-to-0-0-0 out-obj))
                                                   )])
                                             (when code-shift
                                               (let ([new-record (ObjMR
                                                              (smallnoobj-objinfo-obj in-obj000)
                                                              (smallnoobj-objinfo-obj out-obj)
                                                              code-shift
                                                              '("00--0"))])
                                                (set! object-match-list (cons new-record object-match-list))
                                                (set! object-match-listall (cons new-record object-match-listall)))
                                              )
                                             code-shift))))))
                              ; (displayln                            (format "\n\n[--------------------------------DEBUG] Done object-match-list for this pair. param-records => ~s"
                              ;       object-match-list))
                              ; (display-param-records object-match-list)

                               (if param-success?
                                   (cons (ParamMatchRecord out-param object-match-list)
                                         acc-params)
                                   acc-params))])
                          ; (displayln                            (format "\n\n[---------------------------local-param-records-----DEBUG] Done param-combinations for this pair. param-records => ~s"
                          ;           local-param-records))
                          (display-param-records local-param-records)
                        local-param-records))

                    (define this-pair-success? (not (null? param-match-records)))
                    ; (displayln                            (format "[--------------------------------DEBUG] Done param-combinations for this pair. param-records => ~s"
                    ;                 param-match-records))
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

  (send managerid print-all-ids)
  (display-param-records object-match-listall)

  ;; 返回 (all-succeeded? globalParamAnalysis) 仅作演示
  (values all-succeeded? "pass"))

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
; (define dir "/Users/zhangdexiang/github/VSAHDC/arc-dsl/rkt/training-data")

(module+ main
  ; (command-line
  ;  #:args (dir)
  ;  "Usage: racket your-file.rkt <dir>"
   (main dir))
