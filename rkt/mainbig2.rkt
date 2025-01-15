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

  (if result
      (begin
        (displayln "Solution found!")
        (displayln (model result)))
      (displayln "No solution...")))













;; 6.2) 主函数：读取目录 => 对 JSON => 取出第一个 train pair => 合成
(define (main dir)
;;; (displayln "1 info!")
  (define all-json (read-all-json-files dir))  ;; => list of JSON data
  ;;; (displayln "1 info!")
  (for ([json-data (in-list all-json)])
    (displayln "======================================")
    ;;; (displayln (format "Now process JSON: ~a" json-data))
    (displayln (format "Processing file: ~a" (hash-ref json-data 'filename)))

    (define train-data (hash-ref json-data 'train))
    (define test-data  (hash-ref json-data 'test))

    ;; 只示范：拿第一个 train pair
    (define first-pair (if (null? train-data) #f (car train-data)))
    (when first-pair
      (define input-grid  (Grid (hash-ref first-pair 'input)))
      (define output-grid (Grid (hash-ref first-pair 'output)))

      (define input-objects  (objects input-grid  #f #f #f))
      (define output-objects (objects output-grid #f #f #f))

      (define in-obj  (if (set-empty? input-objects)
                          (set)
                          (car (set->list input-objects))))
      (define out-obj (if (set-empty? output-objects)
                          (set)
                          (car (set->list output-objects))))

      (displayln (format "Input obj=~a" in-obj))
      (displayln (format "Output obj=~a" out-obj))

      ;; 调用我们写的合成函数
      (synthesize-transformation in-obj out-obj))))


(provide main)


(module+ main
  ;; 命令行解析
  (command-line
    #:args (dir)
    "Usage: racket your-file.rkt <dir>"
    (main dir)))
