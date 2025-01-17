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
;; 1) 定义一些辅助：object 宽/高，以及各个操作的有效性检查
;; -------------------------------------------------------

;; object-width / object-height 仅作演示，
;; 通过 object-bbox 来拿到 min/max 行列，再计算宽高。
(define (object-width obj)
  (define-values (rmin rmax cmin cmax) (object-bbox obj))
  (if (set-empty? obj)
      0
      (add1 (- rmax rmin)))) ;; 行的范围(含端点) => 宽

(define (object-height obj)
  (define-values (rmin rmax cmin cmax) (object-bbox obj))
  (if (set-empty? obj)
      0
      (add1 (- cmax cmin)))) ;; 列的范围(含端点) => 高

;; 判断 Rot90 的可行性(这里只是示例, 你可换成别的规则)
(define (valid-rot90? obj)
  (and (object? obj)
       (not (set-empty? obj))          ;; 不要空对象
       (> (object-width obj) 1)
       (> (object-height obj) 1)))

;; 判断 HMirror 的可行性
(define (valid-hmirror? obj)
  (and (object? obj)
       (not (set-empty? obj))
       (> (object-width obj) 0)
       ;; 当然你可以添加更多逻辑……
       #t))

;; 判断 VMirror 的可行性
(define (valid-vmirror? obj)
  (and (object? obj)
       (not (set-empty? obj))
       (> (object-height obj) 0)
       #t))

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
(define-symbolic e integer?)

(define (translate e)
  (cond
    [(= e 0) (NoOp)]
    [(= e 1) (Rot90 (NoOp))]
    [(= e 2) (HMirror (NoOp))]
    [(= e 3) (VMirror (NoOp))]
    [else (error "unrecognized transformation code" e)]))

(define (synthesize-transformation input-obj output-obj)

  (if (equal? input-obj output-obj)
    (begin
      (displayln "Solution found => NoOp"  )
      (displayln "#hash((e . 0))")  ;; 或者任何你要输出的信息
      (displayln input-obj )
      'sat)  ;; 函数的返回值

    ;; 否则才做后续的 SMT 求解
    (begin

      (define all-conditions
        (and (>= e 0) (< e 4)  ;; Ensure e is within valid range
            ;; interp 出来的结果必须等于 output-obj
            (equal? (interp (translate e) input-obj)
                    output-obj)))

      (define result (solve (assert all-conditions)))
      (cond
        [(sat? result)
        (displayln "Solution found!  "  )
         (displayln input-obj )
        (displayln (model result))]
        [(unsat? result)
        (displayln " . . . . . . . . ")]
        [else
        (displayln "Unknown result...")]))
    ))

;; -------------------------------------------------------
;; 5) 8 种布尔参数组合 & 汇总生成
;; -------------------------------------------------------
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

(define (objects-with-params grid bools)
  (define b1 (list-ref bools 0))
  (define b2 (list-ref bools 1))
  (define b3 (list-ref bools 2))
  (objects grid b1 b2 b3)) ;; 根据你的实际签名调整

(define (all-objects-from-grid grid)
  (for/fold ([acc (set)])
            ([params (in-list param-combinations)])
    (set-union acc (objects-with-params grid params))))

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
