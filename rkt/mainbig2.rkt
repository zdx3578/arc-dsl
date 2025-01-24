#lang rosette

(require racket/set
         rosette/lib/match
         "objects.rkt"          ;; (objects grid b1 b2 b3)
         "properties.rkt"
         "json-reader.rkt"      ;; (read-all-json-files dir)
         "data-structures.rkt") ;; (struct Grid ...) 等

;; -------------------------------------------------------
;; 1) DSL结构体 (保留 + 新增CMirror, DMirror)
;; -------------------------------------------------------
(struct NoOp    ()          #:transparent)
(struct Rot90   (sub)       #:transparent)
(struct HMirror (sub)       #:transparent)
(struct VMirror (sub)       #:transparent)
(struct CMirror (sub)       #:transparent)  ; NEW
(struct DMirror (sub)       #:transparent)  ; NEW
(struct Compose (e1 e2)     #:transparent)

;; -------------------------------------------------------
;; 2) TransformationInfo 与 transformations 表
;;    在这里添加/删除变换即可
;; -------------------------------------------------------
(struct TransformationInfo (name code apply-fn check-fn dsl-maker)
  #:transparent)

(define transformations
  (list
   (TransformationInfo
    'NoOp
    0
    (lambda (obj) obj)
    (lambda (i o) (equal? i o))
    (lambda (sub) (NoOp)))    ;; sub 暂时用不上, NoOp无子表达式

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
    (lambda (obj) (if (#t) (vmirror obj) #f))
    (lambda (i o) (equal? (vmirror i) o))
    (lambda (sub) (VMirror sub)))

   ;; === 你提到的 CMirror / DMirror，具体实现看需求
   (TransformationInfo
    'CMirror
    4
    (lambda (obj) (cmirror obj)) ; 你需实现 valid-cmirror?, cmirror
    (lambda (i o) (equal? (cmirror i) o))
    (lambda (sub) (CMirror sub)))

   (TransformationInfo
    'DMirror
    5
    (lambda (obj)  (dmirror obj) ) ; 你需实现 valid-dmirror?, dmirror
    (lambda (i o) (equal? (dmirror i) o))
    (lambda (sub) (DMirror sub)))
   ))

;; 辅助函数: 查表
(define (lookup-trans-by-name nm)
  (for/first ([tf (in-list transformations)])
    (when (eq? nm (TransformationInfo-name tf))
      tf)))

(define (lookup-trans-by-code c)
  (for/first ([tf (in-list transformations)])
    (when (= c (TransformationInfo-code tf))
      tf)))

;; -------------------------------------------------------
;; 3) 统一 `interp`：用 dsl-op-name 和 dsl-op-sub
;;    来识别操作名与子表达式，减少重复 match
;; -------------------------------------------------------

;; (A) 提取 DSL操作名
(define (dsl-op-name expr)
  (cond
    [(NoOp? expr)    'NoOp]
    [(Rot90? expr)   'Rot90]
    [(HMirror? expr) 'HMirror]
    [(VMirror? expr) 'VMirror]
    [(CMirror? expr) 'CMirror]    ;; NEW
    [(DMirror? expr) 'DMirror]    ;; NEW
    [else (error "Unknown DSL op (not Compose or recognized unary op)" expr)]))

;; (B) 提取 DSL操作的子表达式(若无子，则返回 #f)
(define (dsl-op-sub expr)
  (cond
    [(NoOp? expr)    #f]
    [(Rot90? expr)   (Rot90-sub expr)]
    [(HMirror? expr) (HMirror-sub expr)]
    [(VMirror? expr) (VMirror-sub expr)]
    [(CMirror? expr) (CMirror-sub expr)]  ;; NEW
    [(DMirror? expr) (DMirror-sub expr)]  ;; NEW
    [else #f]))


;; (C) 真正的 interp
(define (interp expr obj)
  (match expr
    ;; 如果是 Compose
    [(Compose e1 e2)
     (define r1 (interp e1 obj))
     (if r1 (interp e2 r1) #f)]

    ;; 否则认为是单一操作 (NoOp, Rot90, HMirror, VMirror, CMirror, DMirror)
    [_
     (define op (dsl-op-name expr))        ;; 提取操作名
     (define sub (dsl-op-sub expr))        ;; 取子表达式(可为 #f)
     (if (not sub)
         ;; 如果没有子表达式, 如 (NoOp)
         (let ([tf (lookup-trans-by-name op)])
           (if tf ((TransformationInfo-apply-fn tf) obj) #f))
         ;; 有子表达式, 先 interp sub => sub-out
         (let ([sub-out (interp sub obj)])
           (if sub-out
               (let ([tf (lookup-trans-by-name op)])
                 (if tf ((TransformationInfo-apply-fn tf) sub-out) #f))
               #f)))]))


;; -------------------------------------------------------
;; 4) 合成逻辑(与之前类似，但使用 transformations 表)
;; -------------------------------------------------------

;; 4.1) simple-check => 遍历 transformations, 用 check-fn
(define (simple-check in-obj out-obj)
  (for/or ([tf (in-list transformations)])
    (define cfn (TransformationInfo-check-fn tf))
    (when (and cfn (cfn in-obj out-obj))
      (TransformationInfo-name tf))))

;; 4.2) translate => 根据 e 用 dsl-maker
(define-symbolic e integer?)

(define (translate e)
  (define found (lookup-trans-by-code e))
  (if found
      ;; 给 sub = (NoOp) 作为默认下级
      ((TransformationInfo-dsl-maker found) (NoOp))
      (error "unrecognized transformation code" e)))


;; 4.3) synthesize-transformation
(define (synthesize-transformation input-obj output-obj)
  ;; 1. 简单检测
  (define name-result (simple-check input-obj output-obj))
  (cond
    [(symbol? name-result)
     ;; 找到 tf, 打印 code
     (define tf (lookup-trans-by-name name-result))
     (define c  (TransformationInfo-code tf))
     (displayln (format "#hash((e . ~a))" c))
     #t]
    [else
     ;; SMT
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

;; -------------------------------------------------------
;; 以下保持你原先的 process-single-file / main 逻辑
;; -------------------------------------------------------
(define (process-single-file json-data)
  (define train-data (hash-ref json-data 'train))
  (for/and ([pair (in-list train-data)]) ; 所有 pair 必须成功
    (define input-grid (Grid (hash-ref pair 'input)))
    (define output-grid (Grid (hash-ref pair 'output)))
    (define input-obj-set0 (all-objects-from-grid input-grid))
    (define input-obj-set (all-objects-00-c0-from-objs input-obj-set0))

    (for/or ([out-param (in-list param-combinations)]) ; 存在即成功
      (define out-obj-set0 (objects-with-params output-grid out-param))
      (define out-obj-set (all-objects-00-c0-from-objs out-obj-set0))
      (for/and ([out-obj (in-set out-obj-set)])
        (for/or ([in-obj (in-set input-obj-set)])
          (synthesize-transformation in-obj out-obj))))))

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
