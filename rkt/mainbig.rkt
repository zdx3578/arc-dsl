#lang rosette

;; -------------------------------------------------------
;; 0) 引入我们需要的模块
;; -------------------------------------------------------
(require racket/set
        ;;;  typed/racket
         rosette/lib/match
         "objects.rkt"
         "properties.rkt"
        ;;;  "typed-part.rkt"     ;; 假设里头有 (objects grid ...) (rotate90 obj), (hmirror obj), (vmirror obj) 等
         "json-reader.rkt"       ;; 提供 (read-all-json-files dir)
         "data-structures.rkt")  ;; 包含我们对 (struct Grid ...) 以及 (struct Object ...) 的定义等


(struct NoOp ()          #:transparent)
(struct Rot90 (sub)      #:transparent)
(struct HMirror (sub)    #:transparent)
(struct VMirror (sub)    #:transparent)
(struct Compose (e1 e2)  #:transparent)


(define (interp expr obj)
  (match expr
    [(NoOp)
     obj]
    [(Rot90 sub)
     (rotate90 (interp sub obj))]
    [(HMirror sub)
     (hmirror (interp sub obj))]
    [(VMirror sub)
     (vmirror (interp sub obj))]
    [(Compose e1 e2)
     (let ([r1 (interp e1 obj)])
       (interp e2 r1))]))





(define-symbolic e integer?)

(define (translate e)
  (cond
    [(= e 0) (NoOp)]
    [(= e 1) (Rot90 (NoOp))]
    [(= e 2) (HMirror (NoOp))]
    [(= e 3) (VMirror (NoOp))]
    [else (error "unrecognized transformation code" e)]))

(define (synthesize-transformation input-obj output-obj)


  (define all-conditions
    (and (>= e 0) (< e 4)  ;; Ensure e is within valid range
         (equal? (interp (translate e) input-obj)
                 output-obj)))

  (define result (solve (assert all-conditions)))

  (cond
    [(sat? result)
     (displayln "Solution found!")
     (displayln (model result))]
    [(unsat? result)
     (displayln "No solution...")]
    [else
     (displayln "Unknown result...")]))



(define param-combinations
  (list
   (list #f #f #f)
   (list #f #f #t)
   (list #f #t #f)
   (list #f #t #t)
   (list #t #f #f)
   (list #t #f #t)
   (list #t #t #f)
   (list #t #t #t)))

;; 基于单组参数生成对象
(define (objects-with-params grid bools)
  (define b1 (list-ref bools 0))
  (define b2 (list-ref bools 1))
  (define b3 (list-ref bools 2))
  (objects grid b1 b2 b3)) ;; 根据你的实际签名调整

;; 汇总生成：把 8 组参数的结果合并
(define (all-objects-from-grid grid)
  (for/fold ([acc (set)])
            ([params (in-list param-combinations)])
    (set-union acc (objects-with-params grid params))))



;; 6.2) 主函数：读取目录 => 对 JSON => 取出第一个 train pair => 合成
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
        (displayln (format "  input-obj-set count = ~a" (set-count input-obj-set)))


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
                  (displayln (format "-------- ~a  -   in-obj = ~a"
                                    inner-iter
                                    in-obj))
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
