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




;; 用于最终统计“成功的文件”个数
(define total-successful-files 0)

(define (main dir)
  (define all-json (read-all-json-files dir))  ;; => list of JSON data

  (let ([file-iter 0])
    (for ([json-data (in-list all-json)])
      (set! file-iter (add1 file-iter))

      ;;; ;; 显示文件序号和文件名
      ;;; (displayln "=================================================================")
      ;;; (displayln (format "=================================================================File-Iteration #~a | filename: ~a"
      ;;;                    file-iter
      ;;;                    (hash-ref json-data 'filename)))

      ;; 对该文件，先假设“所有 pair 都成功”
      (define file-all-pairs-success? #t)

      ;; 读取 train/test
      (define train-data (hash-ref json-data 'train))
      (define test-data  (hash-ref json-data 'test))

      (let ([pair-iter 0])
        (for ([pair (in-list train-data)])
          (set! pair-iter (add1 pair-iter))

          (displayln
          (format " =================================================================File-Iteration #~a  ~a ------- Now processing train pair #: ~a"
                   file-iter
                   (hash-ref json-data 'filename)
                   pair-iter))

          ;; 取出 input-grid, output-grid
          (define input-grid  (Grid (hash-ref pair 'input)))
          (define output-grid (Grid (hash-ref pair 'output)))

          ;; -- 1) 取出对象集
          (define input-obj-set (all-objects-from-grid input-grid))
          (define output-obj-set (all-objects-from-grid output-grid))


                  ;; -- 1) 对 output-grid 进行 8 种参数组合 -> 并集
          (define output-obj-set (all-objects-from-grid output-grid))
          (define output-00shapes-set (all-objects-00shape-from-objs output-obj-set))
          ;;; (displayln (format "  output-obj-set count = ~a" (set-count output-obj-set)))
          ;;; (displayln (format "  output-00shapes-set count = ~a" (set-count output-00shapes-set)))
          (define diff1 (set-subtract input-obj-set output-obj-set))
          (define diff2 (set-subtract output-obj-set input-obj-set))
          (define diff (set-subtract diff1 diff2 ))

          ;; 在处理这个 pair 时，尝试 8 种 param
          (define param-found? #f)
          (let ([outer-iter 0])
            (for ([out-param (in-list param-combinations)])
              (unless param-found?
                (set! outer-iter (add1 outer-iter))

                ;; 求出此 param 下的所有 out-obj
                (define out-obj-set (objects-with-params output-grid out-param))

                ;; 先假设此 param 可以搞定所有 out-obj
                (define all-out-obj-solved? #t)

                (let ([mid-iter 0])
                  (for ([out-obj (in-set out-obj-set)])
                    (when all-out-obj-solved?
                      (set! mid-iter (add1 mid-iter))

                      (define found-one? #f)
                      (let ([inner-iter 0])
                        (for ([in-obj (in-set input-obj-set)])
                          (unless found-one?
                            (set! inner-iter (add1 inner-iter))
                            (when (synthesize-transformation in-obj out-obj)
                              (set! found-one? #t))))))

                    ;; 若此 out-obj 全部失败
                    (unless found-one?
                      (set! all-out-obj-solved? #f)))))

                ;; 如果此 param 可以搞定所有 out-obj，则这个 pair 成功
                (when all-out-obj-solved?
                  (displayln (format "====> Param ~a solves all out-obj => This pair (pair #:~a) is success!"
                                     out-param pair-iter))
                  (set! param-found? #t)))))

          ;; 当 8 种 param 全部试完后，若 param-found? 仍是 #f, 说明该 pair 失败
          (unless param-found?
            (displayln (format "xxxx> This pair #:~a fails => no param works!"
                               pair-iter))
            ;; 只要有任何一个 pair 失败，就让“本文件不成功”
            (set! file-all-pairs-success? #f)))))

      ;; 如果该文件的所有 pair 都成功，则对全局成功文件计数 +1
      (when file-all-pairs-success?
        (displayln (format "#### This file (filename: ~a) => all pairs success => count+1!"
                           (hash-ref json-data 'filename)))
        (set! total-successful-files (add1 total-successful-files)))))

  (displayln (format "***** total-successful-files = ~a" total-successful-files)))


(provide main)


(module+ main
  ;; 命令行解析
  (command-line
    #:args (dir)
    "Usage: racket your-file.rkt <dir>"
    (main dir)))
