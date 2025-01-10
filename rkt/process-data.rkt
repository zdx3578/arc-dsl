;; process-data.rkt
#lang rosette

(require "data-structures.rkt"
         "helpers.rkt"
         "objects.rkt"
         "properties.rkt")

(provide process-json-data)

;; 处理单个 JSON 数据，提取对象并计算属性
;; 参数：
;; - json-data: 哈希表，包含 "train" 和 "test" 键
;; 返回：
;; - 无（打印输出）
(define (process-json-data json-data)
  "处理单个 JSON 数据，提取对象并执行验证。"

  ;; 提取训练和测试数据
  (define train-data (hash-ref json-data 'train))
  (define test-data (hash-ref json-data 'test))

  ;; 定义所有8种参数组合
  (define param-combinations
    (list
     (list #t #t #t)
     (list #t #t #f)
     (list #t #f #t)
     (list #t #f #f)
     (list #f #t #t)
     (list #f #t #f)
     (list #f #f #t)
     (list #f #f #f)))

  ;; 处理每个输入输出对
  (define (handle-data-pair data-pair)
    ;; 将输入和输出网格转换为 Grid 结构
    (define input-grid (Grid (map list (hash-ref data-pair 'input))))
    (define output-grid (Grid (map list (hash-ref data-pair 'output))))

    ;; 遍历所有参数组合
    (for ([params param-combinations])
      (define univalued? (first params))
      (define diagonal? (second params))
      (define without-bg? (third params))

      ;; 提取输入和输出对象
      (define input-objects (objects input-grid univalued? diagonal? without-bg?))
      (define output-objects (objects output-grid univalued? diagonal? without-bg?))

      ;; 计算输入对象的属性
      (define input-props
        (map (λ (obj)
               (hash
                'size (object-size obj)
                'center (object-center obj)
                'colorcount (object-colorcount obj)))
             (set->list input-objects)))

      ;; 计算输出对象的属性
      (define output-props
        (map (λ (obj)
               (hash
                'size (object-size obj)
                'center (object-center obj)
                'colorcount (object-colorcount obj)))
             (set->list output-objects)))

      ;; 打印参数组合
      (printf "Parameters: univalued?=~a, diagonal?=~a, without_bg?=~a\n"
              univalued? diagonal? without-bg?)

      ;; 打印输入对象
      (printf "Input Objects:\n")
      (for ([obj (set->list input-objects)])
        (displayln (Object-cells obj)))

      ;; 打印输入属性
      (printf "Input Properties:\n")
      (for ([prop input-props])
        (displayln prop))

      ;; 打印输出对象
      (printf "Output Objects:\n")
      (for ([obj (set->list output-objects)])
        (displayln (Object-cells obj)))

      ;; 打印输出属性
      (printf "Output Properties:\n")
      (for ([prop output-props])
        (displayln prop))

      ;; 分隔符
      (printf "----------------------------------------\n"))))

  ;; 处理所有训练数据
  (printf "Processing Train Data:\n")
  (for ([data-pair train-data])
    (handle-data-pair data-pair))

  ;; 处理所有测试数据
  (printf "Processing Test Data:\n")
  (for ([data-pair test-data])
    (handle-data-pair data-pair)))
