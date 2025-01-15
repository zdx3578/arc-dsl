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




(define (synthesize-transformation input-obj output-obj)
  ;; 把 e 定义成一个只能取 'NoOp 'Rot90 'HMirror 'VMirror 的枚举
  (define DSL-ops (list 'NoOp 'Rot90 'HMirror 'VMirror))

  ;; 2) 定义 e 为一个整型符号变量
  (define-symbolic e integer?)

  ;; 3) 加约束：e 必须在 [0, 3] 之间
  (assert (<= 0 e))
  (assert (<= e (sub1 (length DSL-ops))))

  ;; 4) 你的 translate 函数，用数字 => 具体 DSL 操作
  (define (translate idx)
    (match (list-ref DSL-ops idx)
      ['NoOp    (NoOp)]
      ['Rot90   (Rot90 (NoOp))]
      ['HMirror (HMirror (NoOp))]
      ['VMirror (VMirror (NoOp))]
      [_        (error "unrecognized DSL symbol")]))

  ;; 然后跟你之前类似地去 interp + check ...
  (define all-conditions
    (equal? (interp (translate e) input-obj)
            output-obj))

  (define result
    (solve (assert all-conditions)))

  (if result
      (begin
        (displayln "Solution found!")
        (displayln (model result)))
      (displayln "No solution...")))





;;; (define (synthesize-transformation input-obj output-obj)
;;;   ;; 声明一个符号整数 e
;;;   (define-symbolic e integer?)

;;;   ;; 约束 e 的取值只能在 0..3
;;;   (define (translate i)
;;;     (cond
;;;       [(= i 0) (NoOp)]
;;;       [(= i 1) (Rot90 (NoOp))]
;;;       [(= i 2) (HMirror (NoOp))]
;;;       [(= i 3) (VMirror (NoOp))]
;;;       [else    (error "Out of range! e=" i)]))

;;;   (define all-conditions
;;;     (and (<= 0 e) (<= e 3)
;;;          (equal? (interp (translate e) input-obj)
;;;                  output-obj)))

;;;   (define result (solve (assert all-conditions)))
;;;   (if result
;;;       (begin
;;;         (displayln "Solution found!")
;;;         (displayln (model result)))
;;;       (displayln "No solution...")))






;; 6.2) 主函数：读取目录 => 对 JSON => 取出第一个 train pair => 合成
(define (main dir)
;;; (displayln "1 info!")
  (define all-json (read-all-json-files dir))  ;; => list of JSON data
  ;;; (displayln "1 info!")
  (for ([json-data (in-list all-json)])
    (displayln (format "Processing file: ~a" (hash-ref data 'filename)))
    (displayln "======================================")
    (displayln (format "Processing file: ~a" (hash-ref data 'filename)))
    ;;; (displayln (format "Now process JSON: ~a" json-data))

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
