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


;; -------------------------------------------------------
;; 1) DSL 定义（使用 struct 而非 define-type）
;;    参考第3版的写法，但我们在此扩展为支持: NoOp, Rot90, HMirror, VMirror, Compose
;; -------------------------------------------------------
(struct NoOp ()          #:transparent)
(struct Rot90 (sub)      #:transparent)
(struct HMirror (sub)    #:transparent)
(struct VMirror (sub)    #:transparent)
(struct Compose (e1 e2)  #:transparent)

;; 对应的底层函数 rotate90/hmirror/vmirror 等，请在 objects.rkt 或其他文件中实现
;; 这里暂时示例放一个空壳:
;; (define (rotate90 obj) ...)
;; (define (hmirror obj) ...)
;; (define (vmirror obj) ...)

;; -------------------------------------------------------
;; 2) 解释器：interp(e, obj) => 把 DSL 表达式作用到对象 obj 上
;; -------------------------------------------------------
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
  ;; step1: 定义 e 作为 symbol
  (define-symbolic e symbol?)
  ;; step2: 约束 e ∈ { 'NoOp, 'Rot90, 'HMirror, 'VMirror }
  (define constraints
    (assert (or (eq? e 'NoOp)
                (eq? e 'Rot90)
                (eq? e 'HMirror)
                (eq? e 'VMirror))))
  ;; step3: 另行定义一个 (translate e) => DSL struct
  (define (translate sym)
    (cond
      [(eq? sym 'NoOp)    (NoOp)]
      [(eq? sym 'Rot90)   (Rot90 (NoOp))]   ;; for example
      [(eq? sym 'HMirror) (HMirror (NoOp))]
      [(eq? sym 'VMirror) (VMirror (NoOp))]
      [else (error "unrecognized DSL symbol" sym)]))

  ;; step4: 给 solve 加上 "interp" 的最终约束
  ;;        note: in Rosette 2.x we must pass all constraints to (solve).
  (define constraints2
    (assert (equal? (interp (translate e) input-obj)
                    output-obj)))

  ;; unify them:
  (define all-constraints (and constraints constraints2))

  ;; step5: solve
  (define r (solve all-constraints))
  (if r
      (begin
        (displayln "Solution found!")
        (define m (model r))
        (displayln m))
      (displayln "No solution...")))





;; 6.2) 主函数：读取目录 => 对 JSON => 取出第一个 train pair => 合成
(define (main dir)
(displayln "1 info!")
  (define all-json (read-all-json-files dir))  ;; => list of JSON data
  (displayln "1 info!")
  (for ([json-data (in-list all-json)])
    (displayln "======================================")
    (displayln (format "Now process JSON: ~a" json-data))

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

;; -------------------------------------------------------
;; 7) 程序入口
;;    在命令行运行：
;;      racket this-file.rkt <dir>
;;    Rosette 里也可用 (main "某个路径") 方式直接调用
;; -------------------------------------------------------
(provide main)


(module+ main
  ;; 命令行解析
  (command-line
    #:args (dir)
    "Usage: racket your-file.rkt <dir>"
    (main dir)))
