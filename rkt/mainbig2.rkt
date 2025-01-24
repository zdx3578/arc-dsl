#lang rosette

;; -------------------------------------------------------
;; 0) 引入我们需要的模块
;; -------------------------------------------------------
(require racket/set
         rosette/lib/match
         ;;;  racket/control
         "objects.rkt"          ;; (objects grid b1 b2 b3)
         "properties.rkt"
         "json-reader.rkt"      ;; (read-all-json-files dir)
         "data-structures.rkt") ;; (struct Grid ...) 等


;; -------------------------------------------------------
;; 1) 定义一个 TransformationInfo，用于存储变换信息
;;    - name:     变换名称(如 'NoOp, 'Rot90, ...)
;;    - code:     整型编码(如0,1,2,3,...)
;;    - apply-fn: 给定对象，先做 valid? 检查，再做实际操作 => 若不符则返回 #f
;;    - check-fn: (可选)用于 simple-check 的比较函数, 若 #f 表示无简单检查
;;    - dsl-maker: 生成对应 DSL struct 的函数
;; -------------------------------------------------------
(struct TransformationInfo (name code apply-fn check-fn dsl-maker) #:transparent)


;; -------------------------------------------------------
;; 2) DSL 结构体 (保留原有)
;; -------------------------------------------------------
(struct NoOp ()          #:transparent)
(struct Rot90 (sub)      #:transparent)
(struct HMirror (sub)    #:transparent)
(struct VMirror (sub)    #:transparent)
(struct Compose (e1 e2)  #:transparent)


;; -------------------------------------------------------
;; 2.1) 定义全局变换信息表 transformations
;;      将原先各处的 rotate90, valid-rot90? 等都集中到此
;;      如果将来新增 CMirror，只需在此处添加一行
;; -------------------------------------------------------
(define transformations
  (list
   (TransformationInfo
    'NoOp
    0
    ;; apply-fn: NoOp 不需要检查，直接返回原对象
    (lambda (obj) obj)
    ;; check-fn: 用于 simple-check，若不为空就执行 (check-fn in-obj out-obj)
    ;;           这里就是 (equal? i o)
    (lambda (i o) (equal? i o))
    ;; dsl-maker: 生成 DSL struct (NoOp)
    (lambda (sub) (NoOp)))


   (TransformationInfo
    'Rot90
    1
    ;; apply-fn: 先检查 valid-rot90?，再 rotate90
    (lambda (obj)
      (if (valid-rot90? obj)
          (rotate90 obj)
          #f))
    ;; check-fn: simple-check 用 => (equal? (rotate90 i) o)
    (lambda (i o) (equal? (rotate90 i) o))
    ;; dsl-maker
    (lambda (sub) (Rot90 sub)))


   (TransformationInfo
    'HMirror
    2
    (lambda (obj)
      (if (valid-hmirror? obj)
          (hmirror obj)
          #f))
    (lambda (i o) (equal? (hmirror i) o))
    (lambda (sub) (HMirror sub)))


   (TransformationInfo
    'VMirror
    3
    (lambda (obj)
      (if (valid-vmirror? obj)
          (vmirror obj)
          #f))
    (lambda (i o) (equal? (vmirror i) o))
    (lambda (sub) (VMirror sub)))

   ;; 将来若加 (struct CMirror (sub)), code=4:
   ;; (TransformationInfo
   ;;   'CMirror
   ;;   4
   ;;   (lambda (obj) (if (valid-cmirror? obj) (cmirror obj) #f))
   ;;   (lambda (i o) (equal? (cmirror i) o))
   ;;   (lambda (sub) (CMirror sub)))
   ))


;; ============ 一些辅助函数: 查找 transformations 表 ===============

;; 1) 根据符号名字 (e.g. 'Rot90) 找 TransformationInfo
(define (lookup-trans-by-name nm)
  (for/first ([tf (in-list transformations)])
    (when (eq? nm (TransformationInfo-name tf))
      tf)))

;; 2) 根据整型 code 找 TransformationInfo
(define (lookup-trans-by-code c)
  (for/first ([tf (in-list transformations)])
    (when (= c (TransformationInfo-code tf))
      tf)))


;; -------------------------------------------------------
;; 3) 解释器：在执行每个操作前，先调检查器 (数据驱动版本)
;;    - Compose 与以前保持相同
;;    - 对单操作 (NoOp / Rot90 / ...) 不再手写 valid-xxx?
;;      而是用 transformations 里的 apply-fn 去做
;; -------------------------------------------------------
(define (interp expr obj)
  (match expr
    ;; 保留 Compose 逻辑
    [(Compose e1 e2)
     (define r1 (interp e1 obj))
     (if r1
         (interp e2 r1)
         #f)]

    ;; 单操作：NoOp / Rot90(sub) / HMirror(sub) / VMirror(sub) ...
    ;; 只要 match 到 (struct X (sub)) => X 可以是 NoOp, Rot90, ...
    ;;   1) interp sub => sub-out
    ;;   2) apply transformations 里的 apply-fn
    [(or (NoOp) (Rot90 _) (HMirror _) (VMirror _))  ; 未来新增CMirror也可同理
     => (lambda (op-struct)
          (match op-struct
            [(NoOp)
             ;; NoOp 无 sub, 直接将 obj 原样返回 => apply-fn
             (define tf (lookup-trans-by-name 'NoOp))
             (if tf
                 ((TransformationInfo-apply-fn tf) obj)
                 #f)]

            [(Rot90 sub)
             (define sub-out (interp sub obj))
             (if sub-out
                 (let ([tf (lookup-trans-by-name 'Rot90)])
                   (if tf ((TransformationInfo-apply-fn tf) sub-out) #f))
                 #f)]

            [(HMirror sub)
             (define sub-out (interp sub obj))
             (if sub-out
                 (let ([tf (lookup-trans-by-name 'HMirror)])
                   (if tf ((TransformationInfo-apply-fn tf) sub-out) #f))
                 #f)]

            [(VMirror sub)
             (define sub-out (interp sub obj))
             (if sub-out
                 (let ([tf (lookup-trans-by-name 'VMirror)])
                   (if tf ((TransformationInfo-apply-fn tf) sub-out) #f))
                 #f)]))]))


;; -------------------------------------------------------
;; 4) 合成逻辑：用 transformations 表取代硬编码
;; -------------------------------------------------------

;; 4.1) simple-check: 尝试用 transformations 里的 (check-fn i o)
;;      若找到了 => 返回 (TransformationInfo-name tf)
;;      若没有 => #f
(define (simple-check in-obj out-obj)
  (for/or ([tf (in-list transformations)])
    ;; 若该 tf 的 check-fn 存在(非 #f), 且满足 (check-fn in-obj out-obj)
    ;; 则返回 (TransformationInfo-name tf)
    (define maybe-check (TransformationInfo-check-fn tf))
    (when (and maybe-check (maybe-check in-obj out-obj))
      (TransformationInfo-name tf))))


;; 4.2) translate: 根据 e 查 transformations => 用 dsl-maker 构造 DSL expr
(define-symbolic e integer?)

(define (translate e)
  (define found (lookup-trans-by-code e))
  (if found
      ;; 给 sub = (NoOp) 作为缺省？
      ;; 这样就得到 (Rot90 (NoOp)) 等
      ((TransformationInfo-dsl-maker found) (NoOp))
      (error "unrecognized transformation code" e)))


;; 4.3) synthesize-transformation
;;      - 先用 simple-check => 若成功, 打印 code
;;      - 否则进入 SMT => 同样通过 (translate e) + interp
(define (synthesize-transformation input-obj output-obj)
  ;; 1. 先做一次简单的快速检测
  (define name-result (simple-check input-obj output-obj))

  (cond
    [(symbol? name-result)
     ;; 说明成功匹配某个 transformations
     (define tf
       (lookup-trans-by-name name-result))
     (define code (TransformationInfo-code tf))

     (displayln
      (string-append
       "#hash((e . "
       (number->string code)
       "))"))
     #t]  ;; => 表示匹配成功

    ;; 2. 否则进入 SMT
    [else
     (define all-conditions
       (and (>= e 0)
            (< e (length transformations))   ; 这里若 transformations 有4条 => <4
            (equal? (interp (translate e) input-obj) output-obj)))

     (define result (solve (assert all-conditions)))
     (cond
       [(sat? result)
        (displayln "inoutobj found by SMT!")
        (displayln input-obj)
        (displayln (model result))
        #t]
       [(unsat? result)
        #f]
       [else
        (displayln "SMT result: unknown...")
        #f])]))


;; ===============================================
;; 以下为你原先的 process-single-file / main 逻辑
;; ===============================================
(define (process-single-file json-data)
  (define train-data (hash-ref json-data 'train))
  (for/and ([pair (in-list train-data)]) ; 所有 pair 必须成功
    (define input-grid (Grid (hash-ref pair 'input)))
    (define output-grid (Grid (hash-ref pair 'output)))
    (define input-obj-set0 (all-objects-from-grid input-grid))
    (define input-obj-set (all-objects-00-c0-from-objs input-obj-set0))

    ;; 检查是否存在参数组合满足所有输出对象
    (for/or ([out-param (in-list param-combinations)]) ; 存在即成功
      (define out-obj-set0 (objects-with-params output-grid out-param))
      (define out-obj-set (all-objects-00-c0-from-objs out-obj-set0))
      (for/and ([out-obj (in-set out-obj-set)]) ; 所有 out-obj 必须可解
        (for/or ([in-obj (in-set input-obj-set)]) ; 存在可转换的 in-obj
          (synthesize-transformation in-obj out-obj))))))

(define (process-single-file-logging json-data)
  (define fn (hash-ref json-data 'filename))
  (define success? (process-single-file json-data))
  (if success?
      (displayln (format "[] SUCCESS => ~a" fn))
      (begin
        (displayln (format "[] FAIL    => ~a" fn))
        ;; (sleep 1) ; 如果需要暂停可解注
        ))
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
  ;; 命令行解析
  (command-line
    #:args (dir)
    "Usage: racket your-file.rkt <dir>"
    (main dir)))
