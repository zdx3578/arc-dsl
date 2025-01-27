#lang rosette

(require racket/set
         rosette/lib/match
         "objects.rkt"
         "properties.rkt"
         "json-reader.rkt"
         "data-structures.rkt")

;; ========================================================
;; 1) 统一 DSL 结构 + 构造器宏
;; ========================================================
;; 只有一个 struct: (DSL op maybe-sub)
;;  - op:     symbol, e.g. 'NoOp, 'Rot90, 'CMirror, etc.
;;  - sub:    #f if no sub, or (DSL ...) if has sub
(struct DSL (op sub) #:transparent)

;; 定义一组宏，让写法和原先类似:
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

;; 组合操作
;;  - 这里可以把 Compose(e1, e2) 直接存在 sub 里,
;;    例如 sub = (list e1 e2).
(define-syntax-rule (Compose e1 e2)
  (DSL 'Compose (list e1 e2)))


;; ========================================================
;; 2) TransformationInfo + transformations 表
;; ========================================================
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
    (lambda (obj)  (cmirror obj) )
    (lambda (i o) (equal? (cmirror i) o))
    (lambda (sub) (CMirror sub)))

   (TransformationInfo
    'DMirror
    5
    (lambda (obj) (dmirror obj) )
    (lambda (i o) (equal? (dmirror i) o))
    (lambda (sub) (DMirror sub)))
   ))

(define (lookup-trans-by-name nm)
  (for/first ([tf (in-list transformations)])
    (when (eq? nm (TransformationInfo-name tf))
      tf)))

(define (lookup-trans-by-code c)
  (for/first ([tf (in-list transformations)])
    (when (= c (TransformationInfo-code tf))
      tf)))

;; ========================================================
;; 3) interp: 通过 (DSL 'Compose (list e1 e2)) 等区分
;; ========================================================
(define (interp expr obj)
  (match expr
    ;; 组合操作: (DSL 'Compose (list e1 e2))
    [(DSL 'Compose (list e1 e2))
     (define r1 (interp e1 obj))
     (if r1 (interp e2 r1) #f)]

    ;; 单操作, 没有子表达式: (DSL 'NoOp #f)
    [(DSL op #f)
     (define tf (lookup-trans-by-name op))
     (if tf ((TransformationInfo-apply-fn tf) obj) #f)]

    ;; 单操作, 有一个 sub: (DSL 'Rot90 sub)
    [(DSL op sub)
     (define sub-out (interp sub obj))
     (if sub-out
         (let ([tf (lookup-trans-by-name op)])
           (if tf ((TransformationInfo-apply-fn tf) sub-out) #f))
         #f)]))

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
     #t ]
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

;; ========================================================
;; 以下保留你原先的 process-single-file / main
;; ========================================================
;;; (define (process-single-file json-data)
;;;   (define train-data (hash-ref json-data 'train))
;;;   (for/and ([pair (in-list train-data)])
;;;     (define input-grid (Grid (hash-ref pair 'input)))
;;;     (define output-grid (Grid (hash-ref pair 'output)))
;;;     (define input-obj-set0 (all-objects-from-grid input-grid))
;;;     (define input-obj-set (all-objects-00-c0-from-objs input-obj-set0))

;;;     (for/or ([out-param (in-list param-combinations)])
;;;       (define out-obj-set0 (objects-with-params output-grid out-param))
;;;       (define out-obj-set (all-objects-00-c0-from-objs out-obj-set0))
;;;       (for/and ([out-obj (in-set out-obj-set)])
;;;         (for/or ([in-obj (in-set input-obj-set)])
;;;           (synthesize-transformation (ObjectInfo-obj in-obj) (ObjectInfo-obj out-obj) ))))))

;; 每个对象级成功匹配
(struct ObjectMatchRecord
  (in-obj out-obj transform-code details)
  #:transparent)

;; 针对某个 out-param 的匹配成果，收集所有 (ObjectMatchRecord)
(struct ParamMatchRecord
  (param
   object-matches)  ;; list of ObjectMatchRecord
  #:transparent)

;; 针对一个训练对 (input-grid, output-grid) 的转换集合
(struct PairMatchRecord
  (input-grid
   output-grid
   param-match-records)  ;; list of ParamMatchRecord
  #:transparent)






;; 全局收集：若你想把所有 PairMatchRecord 都扔进这个变量
(define pair-match-records '())

;; 主函数
(define (process-single-file json-data)
  ;; 1) 拿到训练数据
  (define train-data (hash-ref json-data 'train))

  ;; 2) 用 for/fold 一次性处理所有 pair
  (define-values (all-succeeded? collected-pairs)
    (for/fold ([acc-succeeded? #t]          ;; 累计：是否目前都成功
               [acc-pairs      '()])        ;; 累计：PairMatchRecord
              ([pair (in-list train-data)])
      ;; --------------------
      ;;    针对单个 pair 的处理
      ;; --------------------
      (define input-grid  (Grid (hash-ref pair 'input)))
      (define output-grid (Grid (hash-ref pair 'output)))

      ;; 提取 input-object 集合
      (define input-obj-set0  (all-objects-from-grid input-grid))
      (define input-obj-set   (all-objects-00-c0-from-objs input-obj-set0))

      ;; 2.1) 收集所有能匹配成功的 param => 形成 param-records
      (define param-records
        (for/fold ([acc-params '()])
                  ([out-param (in-list param-combinations)])
          (define out-obj-set0 (objects-with-params output-grid out-param))
          (define out-obj-set  (all-objects-00-c0-from-objs out-obj-set0))

          ;; 这里我们想记录对象级匹配, for this param
          (define object-match-list '())

          ;; param 下: “所有 out-obj 必须可解” => for/and
          (define param-success?
            (for/and ([out-obj (in-set out-obj-set)])
              ;; 只要有一个 in-obj 能成功 => for/or => 当前 out-obj 成功
              (for/or ([in-obj (in-set input-obj-set)])
                (let ([ok? (synthesize-transformation
                            (ObjectInfo-obj in-obj)
                            (ObjectInfo-obj out-obj))])
                  (when ok?
                    ;; 把匹配信息记入 object-match-list
                    (set! object-match-list
                          (cons (ObjectMatchRecord in-obj out-obj '??? '())
                                object-match-list)))
                  ok?))))
          (displayln object-match-list)
          ;;; (sleep 2)

          ;; 如果 param 成功 => 加入 param-match-record
          (if param-success?
              (cons (ParamMatchRecord out-param object-match-list) acc-params)
              acc-params)))
      ;;; displayln acc-param
      displayln param-records

      ;; 根据 param-records 是否为空，判断此 pair 是否成功
      (define this-pair-success? (not (null? param-records)))

      ;; 若成功，就构造一个 PairMatchRecord 加入累计
      (values
       (and acc-succeeded? this-pair-success?)        ;; 更新全局成功标志
       (if this-pair-success?
           (cons (PairMatchRecord input-grid output-grid param-records)
                 acc-pairs)
           acc-pairs))))

  displayln
  displayln collected-pairs

  ;; 3) 把本文件处理得到的所有 PairMatchRecord 累加进全局
  (set! pair-match-records (append collected-pairs pair-match-records))

  ;; 4) 最终返回：是否所有 pair 都成功
  all-succeeded?)

;; 测试：
;; (process-single-file some-json-data)
;; => 若所有 pair 都成功匹配，则返回 #t，否则返回 #f
;; 同时所有 PairMatchRecord 已经被放进全局 pair-match-records 里了。



(define (process-single-file-logging json-data)
  (define fn (hash-ref json-data 'filename))
  (define success? (process-single-file json-data))
  (if success?
      (displayln (format "[] SUCCESS => ~a" fn))
      (displayln (format "[] FAIL    => ~a" fn)))
  success?)

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
