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


;;; (define (interp expr in-obj)
;;;   (match expr
;;;     [(NoOp)
;;;      ;; NoOp 不变，直接返回 in-obj
;;;      in-obj]

;;;     [(Rot90 sub)
;;;      (define sub-out (interp sub in-obj))
;;;      (if sub-out
;;;          (let ([result (rotate90 sub-out)])
;;;            (let ([expected (list (second (shape sub-out))
;;;                                  (first (shape sub-out)))]
;;;                  [actual   (shape result)])
;;;              (if (equal? actual expected)
;;;                  result
;;;                  #f)))
;;;          #f)]

;;;     [(HMirror sub)
;;;      (define sub-out (interp sub in-obj))
;;;      (if sub-out
;;;          (let ([result (hmirror sub-out)])
;;;            ;; hmirror 不会改变 shape
;;;            (if (equal? (shape result) (shape sub-out))
;;;                result
;;;                #f))
;;;          #f)]

;;;     [(VMirror sub)
;;;      (define sub-out (interp sub in-obj))
;;;      (if sub-out
;;;          (let ([result (vmirror sub-out)])
;;;            ;; vmirror 不会改变 shape
;;;            (if (equal? (shape result) (shape sub-out))
;;;                result
;;;                #f))
;;;          #f)]

;;;     [(Compose e1 e2)
;;;      (define r1 (interp e1 in-obj))
;;;      (if r1
;;;          (interp e2 r1)
;;;          #f)]))



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
    ;; 1.1 匹配到了简单变换，直接输出结果
    [check-result
     (displayln (string-append "Solution found => " (symbol->string check-result)))
     ;; 可以根据 check-result 的不同，输出不同的 e 值
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
     'sat]  ;; 函数返回值

    ;; 1.2 否则进入 SMT 求解
    [else
     ;; 如果没有提前返回，则执行后续与 SMT 相关的逻辑
     (define all-conditions
       (and
         (>= e 0)
         (< e 4)
         ;; interp出来的结果必须等于 output-obj
         (equal? (interp (translate e) input-obj) output-obj)))

     (define result (solve (assert all-conditions)))
     (cond
       [(sat? result)
        (displayln "Solution found by SMT!")
        (displayln input-obj)
        (displayln (model result))]

       [(unsat? result)
        (displayln "SMT result: unsat. No solution!")]

       [else
        (displayln "SMT result: unknown...")])]))



;; -------------------------------------------------------
;; 6) 主函数 main
;; -------------------------------------------------------
(define (main dir)
;;; (displayln "1 info!")
  (define all-json (read-all-json-files dir))  ;; => list of JSON data
  ;;; (displayln "1 info!")
  (let ([file-iter 0])
    (for ([json-data (in-list all-json)])
      (set! file-iter (add1 file-iter))

      ;; 显示文件序号和文件名
      (displayln "=================================================================")
      (displayln (format "=================================================================File-Iteration #~a | filename: ~a"
                         file-iter
                         (hash-ref json-data 'filename)))

      ;; 读取 train/test
      (define train-data (hash-ref json-data 'train))
      (define test-data  (hash-ref json-data 'test))


    ;; 只示范：拿第一个 train pair
      (let ([pair-iter 0])
        (for ([pair (in-list train-data)])
          (set! pair-iter (add1 pair-iter))

          (displayln "-----------------------------------------------------------------")
          (displayln (format " =================================================================File-Iteration #~a  ~a ------- Now processing train pair #: ~a" file-iter (hash-ref json-data 'filename) pair-iter))

          ;; 取出 input-grid, output-grid
          (define input-grid  (Grid (hash-ref pair 'input)))
          (define output-grid (Grid (hash-ref pair 'output)))


        ;; -- 1) 对 input-grid 进行 8 种参数组合 -> 并集
        (define input-obj-set (all-objects-from-grid input-grid))
        (define input-00shapes-set (all-objects-00shape-from-objs input-obj-set))
        (displayln (format "  input-obj-set count = ~a" (set-count input-obj-set)))
        (displayln (format "  input-00shapes-set count = ~a" (set-count input-00shapes-set)))


        ;; -- 2) 现在对 output-grid 的 8 种组合分别处理
        ;;; (displayln "=========================Output=========================")
        (let ([outer-iter 0])
        (for ([out-param (in-list param-combinations)])
          ;; 每进一次循环，计数 + 1
          (set! outer-iter (add1 outer-iter))

          (define out-obj-set (objects-with-params output-grid out-param))
          (displayln "  ")
          (displayln "  ")
          (displayln (format "-----------------------------------------File #~a ~a train pair #: ~a-------------outobj param-iteration ~a---- Output param = ~a, count = ~a"
                            file-iter (hash-ref json-data 'filename) pair-iter outer-iter
                            out-param
                            (set-count out-obj-set)))

          ;; 中层循环：同理，定义一个 mid-iter 计数器
          (let ([mid-iter 0])
            (for ([out-obj (in-set out-obj-set)])
              (set! mid-iter (add1 mid-iter))
              (displayln "  ")
              (displayln "  ")
              (displayln (format "------------------File #~a -- train pair #: ~a--outobj param- ~a--------------out-obj iteration ~a------ Checking out-obj = ~a"
                                file-iter pair-iter outer-iter mid-iter
                                out-obj))

              ;; 最内层循环：再定义一个 inner-iter 计数器
              (let ([inner-iter 0])
                (for ([in-obj (in-set input-obj-set)])
                  (set! inner-iter (add1 inner-iter))
                  ;;; (displayln (format "-------- ~a  -   in-obj = ~a"
                  ;;;                   inner-iter
                  ;;;                   in-obj))
                  (synthesize-transformation in-obj out-obj))))))))))
      )

    (displayln "Done!"))

(provide main)


(module+ main
  ;; 命令行解析
  (command-line
    #:args (dir)
    "Usage: racket your-file.rkt <dir>"
    (main dir)))
