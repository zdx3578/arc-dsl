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
    (lambda (obj)  (rotate90-info obj) )
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
    (lambda (obj)  (vmirror-info obj) )
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

; (define (lookup-trans-by-name nm)
;   (for/first ([tf (in-list transformations)])
;     (when (eq? nm (TransformationInfo-name tf))
;       tf)))
; (define (lookup-trans-by-name nm)
;   (for/first ([tf (in-list transformations)])
;     (and (eq? nm (TransformationInfo-name tf))
;          tf)))

(define (lookup-trans-by-name nm)
  (let ([found (filter (lambda (tf)
                         (eq? nm (TransformationInfo-name tf)))
                       transformations)])
    (if (null? found)
        #f
        (car found))))





;; 统一 apply-op: code => transformations
;; 同时支持 integer/symbol/以及list-of-symbols
(define (apply-op code-or-codes obj)
  ; (displayln (format "apply-op: code-or-codes => ~a" code-or-codes))
  ; (displayln (format "apply-op: obj => ~a" obj))
  (cond
    ;; 若是整数 => lookup-trans-by-code
    [(integer? code-or-codes)
     (define tf (lookup-trans-by-code code-or-codes))
     (if tf ((TransformationInfo-apply-fn tf) obj) #f)]

    ;; 若是符号 => lookup-trans-by-name
    [(symbol? code-or-codes)
     (define tf (lookup-trans-by-name code-or-codes))
     (if tf ((TransformationInfo-apply-fn tf) obj) #f)]

    ;; 若是列表(符号集合) => 依次 apply
    [(and (list? code-or-codes)
          (for/and ([c (in-list code-or-codes)]) (symbol? c)))
     (for/fold ([acc obj])
               ([c (in-list code-or-codes)])
       (define tf (lookup-trans-by-name c))
      ;  (displayln "DEBUG: transformations => ")
      ;   (displayln transformations)
      ;   (for ([tf (in-list transformations)])
      ;     (displayln (format "   transformation name=~a code=~a"
      ;                       (TransformationInfo-name tf)
      ;                       (TransformationInfo-code tf))))
      ;  (displayln (format "--------------apply-op: code-or-codes => ~a" c))
      ;  (displayln (format "apply-op: tf => ~a" tf))
       (if tf ((TransformationInfo-apply-fn tf) acc)
               acc)) ;; 失败则维持原样 or #f, 看需求
     ]

    [else
     (error "apply-op: unexpected code/codes" code-or-codes)]))


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
  (if tf
      (TransformationInfo-code tf)
      #f))

;; 原先的 interp
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
    ['diagonal? (equal? (ObjectInfo-ismove000 obj-info) val)]
    ['univalued? (equal? (ObjectInfo-ismove000 obj-info) val)]
    [_ #f]))

;; 解释 DSLCond
(define (interp-DSLCond dsl obj-info)
  (match dsl
    ;; Base => single transform code
    [(DSLCond 'Base code #f #f #f) (apply-op code obj-info)]

    ;; If => if cond => subT else => subF
    [(DSLCond 'If #f cond subT subF)
     (if (interp-condition cond obj-info)
         (interp-DSLCond subT obj-info)
         (interp-DSLCond subF obj-info))]
    [_ #f]))



(define (simple-check in-obj out-obj)
  (define successful-transformations ; 在外部定义累积列表
    ; 初始值是空列表
    (for/fold ([result '()]) ([tf (in-list transformations)])
      (define cfn (TransformationInfo-check-fn tf))
      (if (and cfn (cfn in-obj out-obj))
          (cons (TransformationInfo-name tf) result) ; 如果转换成功，添加到列表前面
          result))) ; 如果转换失败，则继续原列表

  (begin
    ;;; (displayln (format "All--------------------- successful transformations: ~a" successful-transformations))
    successful-transformations)) ; 返回所有成功的转换记录

(define-symbolic e integer?)

(define (translate e)
  (define found (lookup-trans-by-code e))
  (if found
      ((TransformationInfo-dsl-maker found) (NoOp)) ; default sub => NoOp
      (error "unrecognized transformation code" e)))

(define (synthesize-transformation input-obj output-obj)
  (define name-result (simple-check input-obj output-obj))
  (cond
    [(not (empty? name-result))
     ;;;  (define tf (lookup-trans-by-name name-result))
     ;;;  (define c (TransformationInfo-code tf))
     (displayln (format "#hash((e . ~a))" name-result))
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
     (for/fold ([best (car lst)]) ([x (in-list (cdr lst))])
       (if (pred x best) x best))]))



;; ---------------------------------------------------------------------
;; 4) 统计分析: 在 ParamMatchRecord 层面统计 (diagonal? => transform-code)
;; ---------------------------------------------------------------------
(struct ObjectMatchRecord (in-obj out-obj transform-code details) #:transparent)
(struct ParamMatchRecord (param object-matches) #:transparent)
(struct PairMatchRecord (input-grid output-grid param-match-records) #:transparent)

(define (analyze-param-match-record pmr)
  ;; pmr: (ParamMatchRecord param object-matches)
  ;; 返回一个 hash: key=transformName, val=出现次数
  (define omrs (ParamMatchRecord-object-matches pmr))
  (define transform-count (make-hash))

  ;; 遍历所有 ObjectMatchRecord
  (for ([omr (in-list omrs)])
    (define tcode-list (ObjectMatchRecord-transform-code omr)) ;; 现在是一个列表
    ;; 遍历 transform-code 列表里的每个单独变换
    (for ([single-code (in-list tcode-list)])
      (hash-update! transform-count single-code (λ (old) (add1 old)) 0))) ;; 若不存在旧值，初始为0，然后加1

  transform-count)


(define (analyze-pair-match-record pmRec)
  (define pmrs (PairMatchRecord-param-match-records pmRec))
  ;; acc = 全局 hash: key=transformName, val=累计出现次数
  (for/fold ([acc (make-hash)]) ([p (in-list pmrs)])
    (define local-hash (analyze-param-match-record p))
    ;; 合并 local-hash 到 acc
    (for ([k (in-hash-keys local-hash)])
      (define val (hash-ref local-hash k))
      (hash-update! acc k (λ (old) (+ old val)) 0))
    acc))

(define (build-topN-transform-list transform-hash topN)
  ;; transform-hash: key=transformName, val=出现次数
  ;; 返回出现次数最多的前 topN 个变换的列表
  (define pairs
    (for/list ([k (in-hash-keys transform-hash)])
      (cons k (hash-ref transform-hash k)))) ;; => '((HMirror . 9) (CMirror . 5) ...)
  (define sorted
    (sort pairs (lambda (a b) (> (cdr a) (cdr b))))) ;; 按出现次数从大到小排序

  ;; 可能 sorted 的长度 < topN => 用 (take sorted topN) 会报错
  (define length-sorted (length sorted))
  (define actual-count (min topN length-sorted))

  ;; 若你想若数量不足，就仅返回所有
  (define topN-list (map car (take sorted actual-count)))
  topN-list)



;; ---------------------------------------------------------------------
;; 6) “后处理”阶段：对 pair-match-records 分析 & 构造 if-rule & 测试
;; ---------------------------------------------------------------------
;; 假设仅做一个简单的统计 => 生成一个 candidate-rule
;; 再验证 candidate-rule 在 pair-match-records 里是否都能成功
(define (post-process-rules! pmrs)
  (define transform-hash (make-hash))
  ;; 先把 pmrs 里所有的对象匹配情况汇总
  (for ([pmr (in-list pmrs)])
    (define local-hash (analyze-pair-match-record pmr))
    ;; 合并到 transform-hash
    (for ([k (in-hash-keys local-hash)])
      (define val (hash-ref local-hash k))
      (hash-update! transform-hash k (λ (old) (+ old val)) 0)))

  (define top3 (build-topN-transform-list transform-hash 3))
  (displayln (format "Top 3 transforms => ~s" top3))

  ;; 你可以把这个列表 top3 变换直接构造成一个 DSLCond
  ;; 如果你已经不需要做 if-else，就可以省略 DSLCond 里的 'If
  ;; 以下示例，仅放在 (DSLCond 'Base top3 #f #f #f)
  (define candidate-rule
    (DSLCond 'Base
             top3   ;; 这里就把前3变换放进来
             #f #f #f))

  (displayln (format "Generated rule => ~s" candidate-rule))

  ;; 然后像以前一样，遍历 pmrs 做验证
  (define success?
    (for/and ([pmr (in-list pmrs)])
      (for/and ([param-rec (in-list (PairMatchRecord-param-match-records pmr))])
        (define omrs (ParamMatchRecord-object-matches param-rec))
        (for/and ([omr (in-list omrs)])
          (define in-obj-info (ObjectMatchRecord-in-obj omr))
          (define out-obj (ObjectMatchRecord-out-obj omr))
          ; (displayln (format "in-obj-info => ~s" in-obj-info))
          ; (displayln (format "out-obj => ~s" out-obj))
          (equal? (interp-DSLCond candidate-rule in-obj-info) out-obj)))))

  (displayln (format "Check candidate-rule success? ~a" success?))
  candidate-rule)


;; 全局收集
(define pair-match-records '())

(define (process-single-file json-data)
  ;; 1) 读取训练数据
  (define train-data (hash-ref json-data 'train))

  ;; 2) 用一个 let 包裹外层 for/fold，捕获其多值结果以便打印
  (define-values (all-succeeded? collected-pairs)
    (let-values
        ([(res-succeeded? res-pairs)
          ;; 收集的 PairMatchRecord
          (for/fold ([acc-succeeded? #t] ;; 到目前为止是否全部成功
                     [acc-pairs '()])
                    ([pair (in-list train-data)])
            ;; ---------------------------------------
            ;;   针对单个 pair 的处理
            ;; ---------------------------------------
            (define input-grid (Grid (hash-ref pair 'input)))
            (define output-grid (Grid (hash-ref pair 'output)))

            ;; 提取 input-obj
            (define input-obj-set (all-objects-from-grid input-grid))
            ;;;  (define input-obj-set  (all-objects-00-c0-from-objs input-obj-set0))

            ;; 内层 for/fold: 收集所有能匹配成功的 param => param-records
            (define param-records
              (let ([local-param-records
                     (for/fold ([acc-params '()]) ([out-param (in-list param-combinations)])
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
                                    (let ([code-regular (synthesize-transformation
                                                          in-obj
                                                          out-obj)])
                                      (when code-regular
                                        (set!
                                         object-match-list
                                         (cons (ObjectMatchRecord in-obj out-obj code-regular '("-----0-----" #f))
                                               object-match-list)))
                                      code-regular))])
                             ;; 第 2 步：如果上面那一步 found-regular? 为 #f，就再尝试 shift 匹配
                             (or found-regular?
                                 (for/or ([in-obj (in-set input-obj-set)])
                                   (let ([code-shift (synthesize-transformation
                                                      ( shift-obj-to-0-0-0 in-obj)
                                                      ( shift-obj-to-0-0-0 out-obj))])
                                     (when code-shift
                                       (set! object-match-list
                                             (cons (ObjectMatchRecord in-obj out-obj code-shift '("-----0-----" #t))
                                                   object-match-list)))
                                     code-shift))))))
                       ;; 如果 param-success? => 新增一个 ParamMatchRecord
                       (if param-success?
                           (cons (ParamMatchRecord out-param object-match-list) acc-params)
                           acc-params))])
                ;; ★ 在内层 for/fold 结束后输出调试日志
                (displayln
                 (format "[DEBUG] Done param-combinations for this pair. param-records => ~s"
                         local-param-records))
                local-param-records))
            ;; 判断该 pair 是否成功
            (define this-pair-success? (not (null? param-records)))

            ;; 构造外层新的累积状态
            (define new-succeeded? (and acc-succeeded? this-pair-success?))
            (define new-pairs
              (if this-pair-success?
                  (cons (PairMatchRecord input-grid output-grid param-records) acc-pairs)
                  acc-pairs))

            ;; ★ 在外层 for/fold 这一轮迭代结束前输出调试日志
            (displayln
             (format "[DEBUG] after handling ONE pair => success?=~a, total-collected-pairs=~a"
                     this-pair-success?
                     (length new-pairs)))

            (values new-succeeded? new-pairs))]) ;; 结束 for/fold

      ;; ★ for/fold 全部结束后再打印一次整体结果
      (displayln (format "[DEBUG] all pairs processed => all-succeeded?=~a, total=~a"
                         res-succeeded?
                         (length res-pairs)))
      (values res-succeeded? res-pairs)))

  ;; 3) 把本文件处理的 PairMatchRecord 累加到全局
  (set! pair-match-records (append collected-pairs pair-match-records))

  ;; ★ 显示一下最终的 pair-match-records
  (displayln (format "[DEBUG] appended => pair-match-records total=~a" (length pair-match-records)))
  ;; 3) 做后处理: 生成 if-rule / 统计
  (define candidate-rule (post-process-rules! pair-match-records))

  ;; 4) 若有 test 数据则验证
  (define test-data (hash-ref json-data 'test #f))
  (define test-success?
    (if test-data
        (verify-test-data test-data candidate-rule) ;; 上面示例
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
  (set! pair-match-records '()) ;; 清空全局记录   下一个文件处理的时候是空状态
  success?)

;; 假设 candidate-rule 是一个 DSLCond
(define (apply-if-rule in-grid rule)
  ;; 拿到 in-grid 的所有 objs，或者根据需要处理
  ;; 这里只示意：针对每个对象 interpret 后构造一个新的输出网格
  ;; 实际的实现取决于你自己的结构
  #f) ;; TODO: 你要自行实现

(define (verify-test-data test-data candidate-rule)
  (for/and ([td (in-list test-data)])
    (define in-grid (Grid (hash-ref td 'input)))
    (define out-grid (Grid (hash-ref td 'output))) ;; 真值
    (define predicted (apply-if-rule in-grid candidate-rule))
    (equal? predicted out-grid))) ;; 看你如何定义 equals

(define (main dir)
  (define all-json (read-all-json-files dir))
  (define total-success
    (for/sum ([json-data (in-list all-json)]) (if (process-single-file-logging json-data) 1 0)))

  (displayln (format "[] total-successful-files = ~a" total-success)))

(provide main)

(module+ main
  (command-line #:args (dir) "Usage: racket your-file.rkt <dir>" (main dir)))
