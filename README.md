# Introduction

โปรเจกต์นี้เกี่ยวกับ "CPP Plagiarism Detection" มีวัตถุประสงค์เพื่อสร้างแบบจำลองที่ตรวจจับการลอกเลียนแบบระหว่าง Source Code C++ 2 ชุด โดยพิจารณาจาก Similarity Score และโครงสร้าง Structural Similarity

แกนหลักของระบบคือการใช้ CodeBERT ที่ผ่านการ Fine-tuning บนชุดข้อมูลเฉพาะ ด้วย Architecture แบบ Siamese Network เพื่อเรียนรู้การสร้าง Embedding Vector สำหรับโค้ด C++ ที่สื่อถึงความหมายของโค้ด จากนั้นใช้ Cosine Similarity วัดความคล้ายคลึงและตัดสินผล นอกจากนี้ยังใช้ Tree-sitter วิเคราะห์โค้ดผ่าน Abstract Syntax Tree (AST) ซึ่งเป็นแนวทางที่มีประสิทธิภาพ

# Methodology

### **System Pipeline Overview**

กระบวนการทั้งหมดของระบบแบ่งออกเป็นขั้นตอนหลัก ๆ ดังนี้:

1. **การเตรียมข้อมูล (Data Preparation):**
    - **1.1 โหลดชุดข้อมูลตั้งต้น:** ใช้ชุดข้อมูล `POJ-104` จาก `code_x_glue` ของ Google
    - **1.2 สร้างชุดข้อมูลแบบจับคู่ (Pairing Strategy):** สร้างคู่โค้ดจำนวน 10,000 คู่ ประกอบด้วยคู่ที่ลอกเลียน (Clone) และไม่ลอกเลียน (Non-clone) อย่างละครึ่ง
    - **1.3 แบ่งชุดข้อมูล:** แบ่งข้อมูลที่สร้างขึ้นเป็นชุด Train (70%), Validation (15%), และ Test (15%)
2. **การประมวลผลล่วงหน้า (Preprocessing):**
    - **2.1 การปรับมาตรฐานโค้ด (Code Normalization):** ลบคอมเมนต์, จัดการ Whitespace, และแปลง Keywords เป็นตัวพิมพ์เล็ก เพื่อลดความแตกต่างที่ไม่ใช่สาระสำคัญ
    - **2.1 การสกัด Abstract Syntax Tree (AST):** ใช้ `Tree-sitter` เพื่อแปลงโค้ด C++ เป็นโครงสร้างแบบ Tree และสกัด Node Types ออกมาเป็น Feature
3. **การสร้างและฝึกสอนโมเดล (Model Training):**
    - **3.1 สถาปัตยกรรมโมเดล:** ใช้สถาปัตยกรรมแบบ Siamese Network โดยมี `microsoft/codebert-base` ที่แชร์กันเป็น Encoder
    - **3.2 การ Fine-tuning:** ฝึกสอนโมเดลบนชุดข้อมูลที่จับคู่ไว้ เพื่อให้ CodeBERT เรียนรู้ที่จะสร้าง Embedding Vector สำหรับเปรียบเทียบความหมายของโค้ด C++
4. **การตรวจจับและประเมินผล (Detection & Evaluation):**
    - **4.1 การสร้าง Embeddings:** นำโมเดล ที่ Fine-tune แล้ว มาสร้าง Embedding Vector จากโค้ดในชุด Train/Validation/Test
    - **4.2 การคำนวณความคล้ายคลึง:** ใช้ **Cosine Similarity** เพื่อคำนวณคะแนนความคล้ายคลึงระหว่าง Embedding ของโค้ดแต่ละคู่
    - **4.3 การหาค่า Threshold ที่เหมาะสม:** ใช้ชุด Validation เพื่อหาค่าเกณฑ์ตัดสิน (Threshold) ที่ให้ค่า F1-Score สูงที่สุด
    - **4.4 การประเมินผล:** ประเมินประสิทธิภาพของระบบบนชุด Test โดยใช้ Threshold ที่หาได้ และรายงานผลด้วยเมตริกต่าง ๆ (Accuracy, Precision, Recall, F1-Score, AUC)

# Setting Environment and Dependencies

โปรเจกต์นี้ต้องการไลบรารีหลักดังต่อไปนี้เพื่อการทำงาน:

- **Machine Learning & Data:** `torch`, `transformers`, `datasets`, `scikit-learn`
- **Data Handling:** `numpy`, `pandas`
- **AST Parsing:** `tree-sitter==0.20.4`
- **Utilities:** `matplotlib`, `seaborn`, `tqdm`

# Data Preparation

### **ชุดข้อมูลตั้งต้น (Source Dataset)**

- ใช้ชุดข้อมูล **POJ-104** ซึ่งเป็นส่วนหนึ่งของ **CodeXGLUE benchmark** (`google/code_x_glue_cc_clone_detection_poj104`)
- ชุดข้อมูลนี้ประกอบด้วยโซลูชันของโจทย์ปัญหาโปรแกรมมิ่ง 104 ข้อ แต่ละตัวอย่างข้อมูลประกอบด้วย `id`, `code` (ซอร์สโค้ด C++), และ `label` (หมายเลขของโจทย์ปัญหา)

### **กลยุทธ์การสร้างชุดข้อมูลแบบจับคู่ (Pairing Strategy)**

เนื่องจากชุดข้อมูลเดิมไม่ได้อยู่ในรูปแบบคู่เปรียบเทียบ จึงต้องสร้างขึ้นมาใหม่จำนวน **10,000 คู่** โดยมีเป้าหมายเพื่อสร้างชุดข้อมูลที่สมดุล:

- **คู่ลอกเลียน (Clone Pairs, Label=1):** 5,000 คู่
    - **นิยาม:** โค้ดสองชุดที่ถูกเขียนขึ้นเพื่อแก้โจทย์ปัญหา *ข้อเดียวกัน* จะถือว่าเป็น Clone ในเชิงความหมาย
    - **วิธีการสร้าง:** สุ่มเลือกโค้ดสองชิ้นจากโจทย์ข้อเดียวกัน (มี `label` เหมือนกัน) มาจับคู่กัน
- **คู่ไม่ลอกเลียน (Non-clone Pairs, Label=0):** 5,000 คู่
    - **นิยาม:** โค้ดสองชุดที่ถูกเขียนขึ้นเพื่อแก้โจทย์ปัญหา *คนละข้อ* จะถือว่าไม่เป็น Clone
    - **วิธีการสร้าง:** สุ่มเลือกโจทย์มาสองข้อที่แตกต่างกัน แล้วสุ่มเลือกโค้ดจากแต่ละโจทย์มาจับคู่กัน

*กระบวนการนี้ทำให้มั่นใจได้ว่าโมเดลจะเรียนรู้จากความคล้ายคลึงทางความหมายของอัลกอริทึม ไม่ใช่แค่โครงสร้างปลีกย่อยของโค้ด*

### **การแบ่งชุดข้อมูล (Data Splitting)**

หลังจากสร้างชุดข้อมูล 10,000 คู่แล้ว จะทำการแบ่งข้อมูลออกเป็น 3 ส่วน โดยใช้ `stratify=df['label']` เพื่อรักษาสัดส่วน 50/50 ของข้อมูลแต่ละประเภทไว้ในทุกชุด:

- **Train Set:** 7,000 คู่ (70%)
- **Validation Set:** 1,500 คู่ (15%)
- **Test Set:** 1,500 คู่ (15%)

# Preprocessing & Feature Engineering

### **การปรับมาตรฐานโค้ด (Code Normalization)**

โค้ดดิบจะถูกนำมาปรับให้เป็นมาตรฐานผ่านฟังก์ชัน `normalize_code` เพื่อลดความแตกต่างที่ไม่ส่งผลต่อความหมายของโปรแกรม:

- ลบคอมเมนต์แบบบรรทัดเดียว (`// ...`) และหลายบรรทัด (`/* ... */`)
- ยุบ Whitespace ที่มีหลายตัวติดกันให้เหลือเพียงตัวเดียว
- แปลง C++ Keywords ที่สำคัญ (เช่น `INT`, `FOR`, `IF`) ให้เป็นตัวพิมพ์เล็กทั้งหมด

Example

```markdown
// Before
#include <iostream>
int main() {
    // Calculate sum
    int   sum = 0;
    FOR(int i=0; i<10; i++)  
        sum += i;
    return sum;
}

// After
#include <iostream> int main() { int sum = 0; for(int i=0; i<10; i++) sum += i; return sum; }
```

### **การสกัด Abstract Syntax Tree (AST)**

- **เครื่องมือ:** ใช้ไลบรารี `tree-sitter` ซึ่งเป็น Parser ที่แข็งแกร่งและสามารถแปลงโค้ดเป็นโครงสร้าง Tree ได้อย่างแม่นยำ
- **กระบวนการ:**
    1. โหลด Grammar ของภาษา C++ สำหรับ `tree-sitter`
    2. ฟังก์ชัน `extract_ast_nodes` จะทำการ Parse โค้ดที่ผ่านการ Normalization แล้ว เพื่อสร้าง AST
    3. ทำการ Traverse ไปตาม Node ต่าง ๆ ใน Tree และเก็บรวบรวมประเภทของ Node (เช่น `function_call`, `for_loop`, `variable_decl`)
- **ผลลัพธ์:** แม้ในโปรเจกต์นี้จะมีการสกัด AST ออกมา แต่ข้อมูลนี้ไม่ได้ถูกใช้เป็น Input โดยตรงให้กับโมเดล `SiameseCodeBERT` ในขั้นตอนการฝึกสอน อย่างไรก็ตาม นี่เป็นขั้นตอนสำคัญสำหรับการวิเคราะห์โครงสร้างโค้ด ซึ่งสามารถนำไปใช้กับโมเดลอื่น ๆ หรือเป็น Feature เสริมได้ในอนาคต

## Manual AST vs Tree-Sitter Library

เปรียบเทียบระหว่างการใช้ Regular Expression สร้าง AST และ ใช้ Library [Tree-Sitter](https://tree-sitter.github.io/tree-sitter/) ซึ่งใช้Parserเพื่อสร้าง AST ตามหลักไวยากรณ์ แล้วเอามาใช้evaluationในการประเมิน Plagiarism Detection โดยได้ผลลัพธ์ดังนี้ 

### Code Sample 1

```jsx
void main() {
    int a[200][200];
    int i, j, row, col, sum;
    scanf("%d %d", &row, &col);
    for(i = 0; i < row; i++) {
        for(j = 0; j < col; j++)
            scanf("%d", &a[i][j]);
    }
    if(col >= row) {
        for(i = 0; i < row; i++) {
            sum = 0;
            for(j = 0; j < col; j++)
                sum += a[i][j];
            printf("%d ", sum);
        }
    } else {
        for(j = 0; j < col; j++) {
            sum = 0;
            for(i = 0; i < row; i++)
                sum += a[i][j];
            printf("%d ", sum);
        }
    }
}
```

- Regex AST
    
    ```cpp
    for_loop_count_5 if_stmt_count_5 else_stmt_count_1 max_nesting_4 function_def_count_1 type_int_count_2 type_void_count_1 op_increment_count_3 op_decrement_count_3 op_less_eq_count_2 op_greater_eq_count_3 array_access_count_5 pointer_ops_count_3 stdio_count_5 statement_count_20 paren_complexity_15
    ```
    
    - Regex AST JSON_FROM
        
        ```cpp
        {
          "metrics": {
            "for_loop_count": 5,
            "if_stmt_count": 5,
            "else_stmt_count": 1,
            "function_def_count": 1,
            "type_int_count": 2,
            "type_void_count": 1,
            "op_increment_count": 3,
            "op_decrement_count": 3,
            "op_less_eq_count": 2,
            "op_greater_eq_count": 3,
            "array_access_count": 5,
            "pointer_ops_count": 3,
            "stdio_count": 5,
            "statement_count": 20
          },
          "complexity": {
            "max_nesting": 4,
            "paren_complexity": 15
          }
        }
        ```
        
- Tree-sitter AST
    
    ```cpp
    ts_translation_unit_count_1 ts_function_definition_count_1 ts_primitive_type_count_3 ts_function_declarator_count_1 ts_identifier_count_10 ts_parameter_list_count_1 ts___count_10 ts___count_10 ts_compound_statement_count_10 ts___count_10 ts_declaration_count_2 ts_array_declarator_count_2 ts___count_10 ts_number_literal_count_10 ts___count_10 ts___count_10 ts___count_10 ts_expression_statement_count_10 ts_call_expression_count_6 ts_argument_list_count_6 ts_string_literal_count_6 ts___count_10 ts_string_content_count_6 ts_pointer_expression_count_3 ts___count_3 ts_for_statement_count_10 ts_for_count_10 ts_assignment_expression_count_10 ts___count_10 ts_binary_expression_count_10 ts___count_4 ts_update_expression_count_10 ts____count_6 ts_subscript_expression_count_10 ts_subscript_argument_list_count_10 ts___count_10 ts_if_statement_count_5 ts_if_count_5 ts_condition_clause_count_5 ts____count_9 ts____count_4 ts___count_8 ts_break_statement_count_4 ts_break_count_4 ts_escape_sequence_count_4 ts____count_2 ts___count_2 ts_else_clause_count_1 ts_else_count_1 ts_max_depth_15 ts_control_for_statement_count_5 ts_control_if_statement_count_5 ts_control_else_clause_count_1 ts_func_function_definition_count_1 ts_func_function_declarator_count_1 ts_func_declaration_count_2 ts_expr_binary_expression_count_10 ts_expr_call_expression_count_6 ts_expr_assignment_expression_count_10 ts_stmt_expression_statement_count_10 ts_stmt_break_statement_count_4 ts_stmt_compound_statement_count_10
    ```
    
    - Tree-sitter AST JSON_FROM
        
        ```cpp
        {
          "tree_sitter_counts": {
            "structure": {
              "translation_unit_count": 1,
              "function_definition_count": 1,
              "function_declarator_count": 1,
              "parameter_list_count": 1,
              "compound_statement_count": 10
            },
            "types_and_declarations": {
              "primitive_type_count": 3,
              "identifier_count": 10,
              "declaration_count": 2,
              "array_declarator_count": 2
            },
            "statements": {
              "expression_statement_count": 10,
              "for_statement_count": 10,
              "if_statement_count": 5,
              "break_statement_count": 4,
              "else_clause_count": 1
            },
            "expressions": {
              "call_expression_count": 6,
              "assignment_expression_count": 10,
              "binary_expression_count": 10,
              "update_expression_count": 10,
              "subscript_expression_count": 10,
              "pointer_expression_count": 3
            },
            "literals_and_arguments": {
              "number_literal_count": 10,
              "string_literal_count": 6,
              "string_content_count": 6,
              "argument_list_count": 6,
              "subscript_argument_list_count": 10
            },
            "tokens_and_keywords": {
              "for_count": 10,
              "if_count": 5,
              "else_count": 1,
              "break_count": 4
            }
          },
          "complexity_metrics": {
            "max_depth": 15,
            "control_flow": {
              "for_statement_count": 5,
              "if_statement_count": 5,
              "else_clause_count": 1
            }
          },
          "semantic_counts": {
            "functions": {
              "function_definition_count": 1,
              "function_declarator_count": 1,
              "declaration_count": 2
            },
            "expressions": {
              "binary_expression_count": 10,
              "call_expression_count": 6,
              "assignment_expression_count": 10
            },
            "statements": {
              "expression_statement_count": 10,
              "break_statement_count": 4,
              "compound_statement_count": 10
            }
          }
        }
        ```
        

### Code Sample 2

```jsx
int main() {
    int k, n, m, num[100][100], sum;
    cin >> n >> m;
    for(int i = 0; i < n; i++) {
        for(int j = 0; j < m; j++)
            cin >> num[i][j];
    }
    for(int i = 0; i < n; i++) {
        sum = 0;
        for(int j = 0; j < m; j++)
            sum += num[i][j];
        cout << sum << " ";
    }
    return 0;
}
```

- Regex AST
    
    ```jsx
    function_def_count_1 for_loop_count_3 if_stmt_count_1 max_nesting_5 return_stmt_count_1 type_int_count_5 op_plus_eq_count_1 op_increment_count_3 op_logical_or_count_3 op_equals_count_3 array_access_count_3 iostream_count_5 statement_count_17 paren_complexity
    ```
    
    - Regex AST JSON FORM
        
        ```jsx
        {
          "metrics": {
            "for_loop_count": 3,
            "if_stmt_count": 1,
            "function_def_count": 1,
            "return_stmt_count": 1,
            "type_int_count": 5,
            "op_plus_eq_count": 1,
            "op_increment_count": 3,
            "op_logical_or_count": 3,
            "op_equals_count": 3,
            "array_access_count": 3,
            "iostream_count": 5,
            "statement_count": 17
          },
          "complexity": {
            "max_nesting": 5,
            "paren_complexity": "count_missing"
          }
        }
        ```
        
- Tree-sitter AST
    
    ```jsx
     ts_translation_unit_count_1 ts_function_definition_count_1 ts_primitive_type_count_5 ts_function_declarator_count_1 ts_identifier_count_10 ts_parameter_list_count_1 ts___count_5 ts___count_5 ts_compound_statement_count_5 ts___count_5 ts_declaration_count_4 ts___count_4 ts_array_declarator_count_2 ts___count_6 ts_number_literal_count_10 ts___count_6 ts___count_10 ts_expression_statement_count_9 ts_binary_expression_count_10 ts____count_4 ts____count_5 ts_for_statement_count_3 ts_for_count_3 ts_init_declarator_count_3 ts___count_4 ts___count_3 ts_update_expression_count_3 ts____count_3 ts_assignment_expression_count_2 ts_subscript_expression_count_4 ts_subscript_argument_list_count_4 ts_string_literal_count_1 ts___count_2 ts_string_content_count_1 ts_if_statement_count_1 ts_if_count_1 ts_condition_clause_count_1 ts____count_4 ts____count_3 ts___count_2 ts____count_1 ts___count_5 ts_return_statement_count_1 ts_return_count_1 ts_max_depth_15 ts_control_for_statement_count_3 ts_control_if_statement_count_1 ts_func_function_definition_count_1 ts_func_function_declarator_count_1 ts_func_declaration_count_4 ts_expr_binary_expression_count_10 ts_expr_assignment_expression_count_2 ts_stmt_expression_statement_count_9 ts_stmt_return_statement_count_1 ts_stmt_compound_statement_count_5
    
    ```
    
    - Tree-sitter AST JSON FORM
        
        ```jsx
        {
          "tree_sitter_counts": {
            "structure": {
              "translation_unit_count": 1,
              "function_definition_count": 1,
              "function_declarator_count": 1,
              "parameter_list_count": 1,
              "compound_statement_count": 5
            },
            "types_and_declarations": {
              "primitive_type_count": 5,
              "identifier_count": 10,
              "declaration_count": 4,
              "array_declarator_count": 2,
              "init_declarator_count": 3
            },
            "statements": {
              "expression_statement_count": 9,
              "for_statement_count": 3,
              "if_statement_count": 1,
              "return_statement_count": 1
            },
            "expressions": {
              "binary_expression_count": 10,
              "update_expression_count": 3,
              "assignment_expression_count": 2,
              "subscript_expression_count": 4
            },
            "literals_and_arguments": {
              "number_literal_count": 10,
              "string_literal_count": 1,
              "string_content_count": 1,
              "subscript_argument_list_count": 4
            },
            "tokens_and_keywords": {
              "for_count": 3,
              "if_count": 1,
              "return_count": 1,
              "condition_clause_count": 1
            }
          },
          "complexity_metrics": {
            "max_depth": 15,
            "control_flow": {
              "for_statement_count": 3,
              "if_statement_count": 1
            }
          },
          "semantic_counts": {
            "functions": {
              "function_definition_count": 1,
              "function_declarator_count": 1,
              "declaration_count": 4
            },
            "expressions": {
              "binary_expression_count": 10,
              "assignment_expression_count": 2
            },
            "statements": {
              "expression_statement_count": 9,
              "return_statement_count": 1,
              "compound_statement_count": 5
            }
          }
        }
        ```
        

| Metric | Manual AST (ใช้ Regex) | Tree-sitter AST |
| --- | --- | --- |
| F1-Score | 0.734 | 0.763 |
| Precision | 0.623 | 0.709 |
| Recall | 0.893 | 0.825 |
| Accuracy | 0.657 | 0.728 |
| Best Threshold | 0.024 | 0.091 |
| False Positives | 430 | 269 (ดีกว่า) |
| False Negatives | 85 (ดีกว่า) | 139 |

จากผลลัพธ์ดังกล่าวเราจึงเลือกใช้Library Tree-sitter เนื่องจากความง่ายต่อการใช้งาน

# Model Architecture and Fine-tuning

### การเลือก Sample และ Problem

![chart.png](attachment:86b927e6-5d24-45cd-a7da-ef100faba915:chart.png)

เราเลือกใช้ 10,000 pairs และ 10 problems เนื่องจากการใช้เพียง 2 problems ให้ผลลัพธ์ไม่ต่างจากการให้โมเดลสุ่มโยนเหรียญเลย

### **โมเดลหลัก (Core Model)**

- ใช้ **CodeBERT** (`microsoft/codebert-base`) ซึ่งเป็นโมเดล Transformer ที่ผ่านการ Pre-train มาบนชุดข้อมูลโค้ดขนาดใหญ่ ทำให้มีความเข้าใจในโครงสร้างและไวยากรณ์ของภาษาโปรแกรมมิ่งเป็นอย่างดี
- คุณสมบัติของ CodeBERT
    - Pre-trained บน 6.4M โค้ดจาก GitHub (6 ภาษาโปรแกรม)
    - เข้าใจทั้ง natural language และ programming language
    - Architecture: 12-layer Transformer, 768 hidden size, 12 attention heads
    - Token vocabulary: ~50,000 tokens

### **Siamese Network Architecture**

โมเดล `SiameseCodeBERT` ถูกสร้างขึ้นเพื่อเปรียบเทียบโค้ดสองชุด:

1. **Shared Encoder:** โมเดล CodeBERT ตัวเดียวกันถูกใช้เป็น Encoder สำหรับโค้ดทั้งสองชุด (code1 และ code2) การแชร์น้ำหนัก (Weight Sharing) ทำให้โมเดลเรียนรู้ที่จะสร้าง Embedding ในปริภูมิ (Space) เดียวกัน
2. **Input:** โค้ดที่ผ่านการ Normalization แล้วจะถูก Tokenize และป้อนเข้าสู่ CodeBERT
3. **Pooling:** Output จากชั้นสุดท้ายของ CodeBERT (Last Hidden State) จะถูกนำมาทำ **Mean Pooling** เพื่อสร้าง Embedding Vector ขนาด 768 มิติ ที่เป็นตัวแทนของโค้ดทั้ง snippet
4. **Similarity Learning:** แทนที่จะใช้ Cosine Similarity โดยตรงในระหว่างการฝึก โมเดลนี้คำนวณ **ค่าผลต่างแบบ Absolute** ระหว่าง Embedding Vector ทั้งสอง แล้วป้อนผลต่างนี้เข้าสู่ Classification Head (ประกอบด้วยชั้น `Linear`, `ReLU`, `Dropout`) เพื่อทำนายว่าเป็นคู่ลอกเลียนหรือไม่ (Output ผ่าน `Sigmoid`)

```markdown
class SiameseCodeBERT(nn.Module):
    def __init__(self, encoder):
        self.encoder = encoder  # CodeBERT (shared)
        self.classifier = nn.Sequential(
            nn.Linear(768, 256),    # Compress embedding
            nn.ReLU(),              # Non-linearity
            nn.Dropout(0.1),        # Regularization
            nn.Linear(256, 1),      # Binary classification
            nn.Sigmoid()            # Output [0, 1]
        )
```

### **Mean Pooling Strategy**

**Mean Pooling** = คือวิธีสร้าง embedding ของ code/text sequence โดยเอาค่าเฉลี่ยของ hidden states ทุก token (หลัง encode ด้วย CodeBERT/Transformer) โดย ถ่วงน้ำหนักด้วย attention mask เพื่อไม่เอา padding มาคิด

### **กระบวนการฝึกสอน (Training Process)**

- **Epochs:** 3
- **Batch Size:** 16
- **Learning Rate:** 2e-5
- **Loss Function:** `Binary Cross-Entropy Loss (BCELoss)` เหมาะสำหรับงาน Binary Classification
- **Optimizer:** `AdamW`
- **Scheduler:** `Linear schedule with warmup` เพื่อช่วยให้การฝึกมีเสถียรภาพ

ผลการฝึกแสดงให้เห็นว่าโมเดลสามารถเรียนรู้ได้อย่างรวดเร็ว โดยมี Accuracy บน Validation Set สูงถึง **97.8%** หลังสิ้นสุด Epoch ที่ 3

# Inference and Evaluation

### **การสร้าง Embeddings**

ในขั้นตอน Inference จะใช้เฉพาะส่วน Encoder ของโมเดล `SiameseCodeBERT` ที่ผ่านการ Fine-tuning แล้ว เพื่อสร้าง Embedding Vector ขนาด 768 มิติสำหรับโค้ดแต่ละชุด

### **การคำนวณความคล้ายคลึง**

- นำ Embedding Vector ที่ได้จากขั้นตอน 7.1 มาทำ **L2 Normalization**
- คำนวณ **Cosine Similarity** ระหว่าง Vector ของโค้ดแต่ละคู่ เพื่อให้ได้คะแนนความคล้ายคลึงในช่วง [-1, 1] (แต่ในทางปฏิบัติมักจะเป็น [0, 1] สำหรับงานนี้)
Similarity=cos(θ)=∥A∥∥B∥ / A⋅B

### **การหาค่า Threshold ที่เหมาะสม**

- การแปลงคะแนน Similarity ให้เป็นการตัดสินใจแบบ Binary (Plagiarism/Original) จำเป็นต้องมีค่า Threshold
- โปรเจกต์นี้ใช้วิธีที่ถูกต้อง โดยการหาค่า Threshold จาก **Validation Set** โดยเลือกค่าที่ทำให้ **F1-Score สูงที่สุด** ซึ่งในที่นี้คือ **0.615**

### **การประเมินผลบน Test Set**

เมื่อได้ Threshold ที่เหมาะสมแล้ว จึงนำไปใช้กับ **Test Set** เพื่อประเมินประสิทธิภาพที่แท้จริงของระบบ ผลลัพธ์ที่ได้คือ:

- **Accuracy:** 93.93%
- **Precision:** 92.52%
- **Recall:** 95.60%
- **F1-Score:** 94.03%
- **AUC:** 98.92%

ค่า F1-Score ที่สูงถึง 94% บน Test Set ยืนยันว่าระบบมีประสิทธิภาพสูงในการตรวจจับการลอกเลียนแบบโค้ด C++
