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
;; 2) DSL 结构体
;; -------------------------------------------------------
(struct NoOp ()          #:transparent)
(struct Rot90 (sub)      #:transparent)
(struct HMirror (sub)    #:transparent)
(struct VMirror (sub)    #:transparent)
(struct Compose (e1 e2)  #:transparent)

;; -------------------------------------------------------
;; 3) 解释器：在执行每个操作前，先调检查器
;; -------------------------------------------------------
(define (interp expr obj)
  (match expr
    [(NoOp)
     obj]

    [(Rot90 sub)
     (define sub-out (interp sub obj))
     (if (valid-rot90? sub-out)
         (rotate90 sub-out)
         #f)]

    [(HMirror sub)
     (define sub-out (interp sub obj))
     (if (valid-hmirror? sub-out)
         (hmirror sub-out)
         #f)]

    [(VMirror sub)
     (define sub-out (interp sub obj))
     (if (valid-vmirror? sub-out)
         (vmirror sub-out)
         #f)]

    [(Compose e1 e2)
     (define r1 (interp e1 obj))
     (if r1
         (interp e2 r1)
         #f)]))





;; -------------------------------------------------------
;; 4) 合成逻辑：我们用整型 e 代替符号，范围是 [0..3]
;; -------------------------------------------------------

(define (simple-check in-obj out-obj)
  (cond
    [(equal? in-obj out-obj)
     'NoOp]
    [(equal? (rotate90 in-obj) out-obj)
     'Rot90]
    [(equal? (hmirror in-obj) out-obj)
     'HMirror]
    [(equal? (vmirror in-obj) out-obj)
     'VMirror]
    [else
     #f]))  ;; #f 表示没找到匹配的变换



(define-symbolic e integer?)

(define (translate e)
  (cond
    [(= e 0) (NoOp)]
    [(= e 1) (Rot90 (NoOp))]
    [(= e 2) (HMirror (NoOp))]
    [(= e 3) (VMirror (NoOp))]
    [else (error "unrecognized transformation code" e)]))

(define (synthesize-transformation input-obj output-obj)
  ;; 1. 先做一次简单的快速检测
  (define check-result (simple-check input-obj output-obj))

  (cond
    ;; 1.1 如果简单检测到变换成功，就返回 #t
    [check-result
     (displayln (string-append "inoutobj found => " (symbol->string check-result)))
     ;; 如果需要，可以打印出不同 e 的值:
     (displayln
      (string-append
       "#hash((e . "
       (case check-result
         [(NoOp)     "0"]
         [(Rot90)    "1"]
         [(HMirror)  "2"]
         [(VMirror)  "3"])
       "))"))
     (displayln input-obj)
     ;; 直接返回 #t 表示成功
     #t]

    ;; 1.2 否则进入 SMT 求解
    [else
     (define all-conditions
       (and
        (>= e 0)
        (< e 4)
        (equal? (interp (translate e) input-obj) output-obj)))

     (define result (solve (assert all-conditions)))
     (cond
       [(sat? result)
        (displayln "inoutobj found by SMT!")
        (displayln input-obj)
        (displayln (model result))
        ;; 表示成功
        #t]

       [(unsat? result)
        (displayln "            .              .             ")
        ;; 表示失败
        #f]

       [else
        (displayln "SMT result: unknown...")
        ;; 也返回 #f，表示暂时无解
        #f])]))


(define total-successful-files 0)


(define (main dir)
  (define all-json (read-all-json-files dir)) ; 假设已定义 read-all-json-files
  (set! total-successful-files
        (for/sum ([json-data (in-list all-json)]) ; 使用 for/sum 直接统计成功文件
          (if (process-single-file json-data) 1 0)))
  (displayln (format "***** total-successful-files = ~a" total-successful-files)))

;; 处理单个文件
(define (process-single-file json-data)
  (define train-data (hash-ref json-data 'train))
  (for/and ([pair (in-list train-data)]) ; 所有 pair 必须成功
    (define input-grid (Grid (hash-ref pair 'input)))
    (define output-grid (Grid (hash-ref pair 'output)))
    (define input-obj-set (all-objects-from-grid input-grid))
    ;;; (define output-obj-setall (all-objects-from-grid output-grid))

    ;; 检查是否存在参数组合满足所有输出对象
    (for/or ([out-param (in-list param-combinations)]) ; 存在即成功
      (define out-obj-set (objects-with-params output-grid out-param))
      (for/and ([out-obj (in-set out-obj-set)]) ; 所有 out-obj 必须可解
        (for/or ([in-obj (in-set input-obj-set)]) ; 存在可转换的 in-obj
          (synthesize-transformation in-obj out-obj))))))

(provide main)


(module+ main
  ;; 命令行解析
  (command-line
    #:args (dir)
    "Usage: racket your-file.rkt <dir>"
    (main dir)))
