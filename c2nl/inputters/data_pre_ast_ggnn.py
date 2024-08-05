import json
import numpy as np
import wordninja
from tqdm import tqdm
from scipy.sparse import coo_matrix

def build_input(ast,n_node,):
    # 初始化用于存储数据流关系的图
    data_flow_graph = {}
    subword_map = {} # word->subword
    type_map = {} # type->word
    node_map = {} # node->type
    type_id = 0
    word_id = 0

    # 遍历AST节点
    ast_token = []
    # 元素为1的位置是AST的边，元素为2的位置是控制流边，元素为3的位置是数据流的边，控制流和数据流都包含了AST的边



    next_siblings = {} # node_id
    leave_id = []
    edges = []
    # data_flow = set()
    # control_flow = set()
    for node in ast:

        type_id = len(ast_token)
        node_map[str(node['id'])] = type_id
        ast_token.append(node["type"])

        # type_map[str(type_id)] = []

        if "value" in node.keys():
            word_id = len(ast_token)
            type_map[str(type_id)] = word_id
            ast_token.append(node["value"])
            subword_map[str(word_id)] = []
            if len(wordninja.split(node['value']))>1:
                for subword in wordninja.split(node['value']):
                    subword_map[str(word_id)].append(len(ast_token))
                    ast_token.append(subword)
        # elif "value" not in node.keys:
        #
        if 'children' in node.keys():
            for i,child in enumerate(node['children'][:-1]):
                next_siblings[str(child)] = node['children'][i+1]

    for node in ast:
        if "children" in node.keys():
            for child in node["children"]:
                edges.append((node_map[str(node['id'])],node_map[str(child)],0))
                # edges.add((node_map[str(child)],node_map[str(node['id'])]))
                if "children" not in ast[child].keys():
                    try:
                        edges.append((type_map[str(node_map[str(node['id'])])],type_map[str(node_map[str(child)])],0))
                        # edges.add((type_map[str(node_map[str(child)])],type_map[str(node_map[str(node['id'])])]))
                    except:
                        pass
                #     print(child)
                #     print(node['id'])
                #     print(node_map)
                #     print(type_map)
                #     print(ast_token)
                #     raise
        else:
            leave_id.extend(subword_map[str(type_map[str(node_map[str(node['id'])])])])
    # 加一条自己到自己的边
    for i, token in enumerate(ast_token):
        edges.append((i,i,0))
    # 给叶子节点加边
    for i in range(len(leave_id)-1):
        edges.append((leave_id[i],leave_id[i+1],1))
        # edges.add((leave_id[i+1],leave_id[i]))


    # add edges between value token and its subtokens
    for key,values in subword_map.items():
        for value in values:
            edges.append((int(key),value, 0))
            # edges.add((value,int(key)))
            # data_flow.add((int(key),value))
    # add edges between type token and its value
    for key,value in type_map.items():
        edges.append((int(key), value,0))
        # edges.add((value,int(key)))


    for node in ast:
        last_child = None
        if "children" in node.keys():
            for child in node["children"]:
                # edges.add((node_map[str(node['id'])],node_map[str(child)]))
                if node["type"] == "ForStatement" and last_child is not None:
                    edges.append((node_map[str(last_child)],node_map[str(child),2]))
                    # edges.add((node_map[str(child)],node_map[str(last_child)]))
                    try:
                        edges.append((node_map[str(node['id'])], node_map[str(next_siblings[str(node['id'])])],2))
                        # edges.add((node_map[str(next_siblings[str(node['id'])])], node_map[node['id']]))
                    except:
                        pass
                elif node["type"] == "BlockStatement" and last_child is not None:
                    edges.append((node_map[str(last_child)],node_map[str(child)],2))
                    # edges.add((node_map[str(child)],node_map[str(last_child)]))
                elif node["type"] == "WhileStatement" and last_child is not None:
                    edges.append((node_map[str(last_child)],node_map[str(child)],2))
                    # edges.add((node_map[str(child)],node_map[str(last_child)]))
                    try:
                        edges.append((node_map[str(node['id'])], node_map[str(next_siblings[str(node['id'])])],2))
                        # edges.add((node_map[str(next_siblings[str(node['id'])])], node_map[node['id']]))
                    except:
                        pass
                elif node["type"] == "IfStatement" and last_child is not None:
                    edges.append((node_map[str(last_child)],node_map[str(child)],2))
                    # edges.add((node_map[str(child)],node_map[str(last_child)]))
                    try:
                        edges.append((node_map[str(node['id'])], node_map[str(next_siblings[str(node['id'])])],2))
                        # edges.add((node_map[str(next_siblings[str(node['id'])])], node_map[node['id']]))
                    except:
                        pass

                last_child = child
                # try:
                if node['type'] == 'SwitchStatement' and ast[child]['type'] == 'SwitchStatementCase' and str(node['id']) in next_siblings and 'children' in ast[child]:
                    for i in ast[child]['children']:
                        if 'BreakStatement' in ast[i]['type']:
                            edges.append((node_map[str(i)], node_map[str(next_siblings[str(node['id'])])],2))
                            # edges.add((node_map[str(next_siblings[str(node['id'])])],node_map[str(i)]))
                # except:
                #     print('node_map:'+str(node_map))
                #     print('node_map[str(i)]:'+str(node_map[str(i)]))
                #     print('next_siblings:'+str(next_siblings))
                #     print("node['id']:"+str(node['id']))
                #     print("i:"+str(i))
                #     raise
    last_use = {}
    for node in ast:
        # 如果是变量定义节点，将其添加到图中
        if node["type"] == "FormalParameter":
            variable_name = node["value"]
            data_flow_graph[variable_name] = type_map[str(node_map[str(node["id"])])]

        # 如果是变量使用节点，将其添加到对应变量的usages中
        elif node["type"] in ["ReferenceType", "MemberReference", "MethodInvocation"]:
            variable_name = node["value"]
            if variable_name in data_flow_graph:
                edge1 = (data_flow_graph[variable_name],type_map[str(node_map[str(node["id"])])],3)
                # edge2 = (type_map[str(node_map[str(node["id"])])],data_flow_graph[variable_name])
                edges.append(edge1)
                # edges.add(edge2)
                last_use[variable_name] = type_map[str(node_map[str(node["id"])])]
            if variable_name in last_use:
                edge1 = (last_use[variable_name],type_map[str(node_map[str(node["id"])])],3)
                # edge2 = (type_map[str(node_map[str(node["id"])])],last_use[variable_name])
                edges.append(edge1)
                # edges.add(edge2)

    return ast_token,coo_matrix(att_matrix)


def create_adjacency_matrix(save_edge_digit_list, n_node, n_edge_types):
    a = np.zeros([n_node, n_node * n_edge_types * 2])

    for edge in save_edge_digit_list:
        src_idx = edge[0]
        tgt_idx = edge[1]
        e_type = edge[2]

        a[tgt_idx][(e_type) * n_node + src_idx] = 1
        a[src_idx][(e_type + n_edge_types) * n_node + tgt_idx] = 1

    return a
# 使用给定的AST示例调用函数
# ast_example = [{"id": 0, "type": "MethodDeclaration", "children": [1, 2, 4, 8, 15, 53, 63, 72, 76, 90, 136, 142], "value": "findPLV"}, {"id": 1, "type": "BasicType", "value": "int"}, {"id": 2, "type": "FormalParameter", "children": [3], "value": "M_PriceList_ID"}, {"id": 3, "type": "BasicType", "value": "int"}, {"id": 4, "type": "LocalVariableDeclaration", "children": [5, 6], "value": "Timestamp"}, {"id": 5, "type": "ReferenceType", "value": "Timestamp"}, {"id": 6, "type": "VariableDeclarator", "children": [7], "value": "priceDate"}, {"id": 7, "type": "Literal", "value": "null"}, {"id": 8, "type": "LocalVariableDeclaration", "children": [9, 10], "value": "String"}, {"id": 9, "type": "ReferenceType", "value": "String"}, {"id": 10, "type": "VariableDeclarator", "children": [11], "value": "dateStr"}, {"id": 11, "type": "MethodInvocation", "children": [12, 13, 14], "value": "Env.getContext"}, {"id": 12, "type": "MethodInvocation", "value": "Env.getCtx"}, {"id": 13, "type": "MemberReference", "value": "p_WindowNo"}, {"id": 14, "type": "Literal", "value": "\"DateOrdered\""}, {"id": 15, "type": "IfStatement", "children": [16, 23, 30], "value": "if"}, {"id": 16, "type": "BinaryOperation", "children": [17, 20]}, {"id": 17, "type": "BinaryOperation", "children": [18, 19]}, {"id": 18, "type": "MemberReference", "value": "dateStr"}, {"id": 19, "type": "Literal", "value": "null"}, {"id": 20, "type": "BinaryOperation", "children": [21, 22]}, {"id": 21, "type": "MethodInvocation", "value": "dateStr.length"}, {"id": 22, "type": "Literal", "value": "0"}, {"id": 23, "type": "StatementExpression", "children": [24], "value": "priceDate"}, {"id": 24, "type": "Assignment", "children": [25, 26]}, {"id": 25, "type": "MemberReference", "value": "priceDate"}, {"id": 26, "type": "MethodInvocation", "children": [27, 28, 29], "value": "Env.getContextAsDate"}, {"id": 27, "type": "MethodInvocation", "value": "Env.getCtx"}, {"id": 28, "type": "MemberReference", "value": "p_WindowNo"}, {"id": 29, "type": "Literal", "value": "\"DateOrdered\""}, {"id": 30, "type": "BlockStatement", "children": [31, 38], "value": "{"}, {"id": 31, "type": "StatementExpression", "children": [32], "value": "dateStr"}, {"id": 32, "type": "Assignment", "children": [33, 34]}, {"id": 33, "type": "MemberReference", "value": "dateStr"}, {"id": 34, "type": "MethodInvocation", "children": [35, 36, 37], "value": "Env.getContext"}, {"id": 35, "type": "MethodInvocation", "value": "Env.getCtx"}, {"id": 36, "type": "MemberReference", "value": "p_WindowNo"}, {"id": 37, "type": "Literal", "value": "\"DateInvoiced\""}, {"id": 38, "type": "IfStatement", "children": [39, 46], "value": "if"}, {"id": 39, "type": "BinaryOperation", "children": [40, 43]}, {"id": 40, "type": "BinaryOperation", "children": [41, 42]}, {"id": 41, "type": "MemberReference", "value": "dateStr"}, {"id": 42, "type": "Literal", "value": "null"}, {"id": 43, "type": "BinaryOperation", "children": [44, 45]}, {"id": 44, "type": "MethodInvocation", "value": "dateStr.length"}, {"id": 45, "type": "Literal", "value": "0"}, {"id": 46, "type": "StatementExpression", "children": [47], "value": "priceDate"}, {"id": 47, "type": "Assignment", "children": [48, 49]}, {"id": 48, "type": "MemberReference", "value": "priceDate"}, {"id": 49, "type": "MethodInvocation", "children": [50, 51, 52], "value": "Env.getContextAsDate"}, {"id": 50, "type": "MethodInvocation", "value": "Env.getCtx"}, {"id": 51, "type": "MemberReference", "value": "p_WindowNo"}, {"id": 52, "type": "Literal", "value": "\"DateInvoiced\""}, {"id": 53, "type": "IfStatement", "children": [54, 57], "value": "if"}, {"id": 54, "type": "BinaryOperation", "children": [55, 56]}, {"id": 55, "type": "MemberReference", "value": "priceDate"}, {"id": 56, "type": "Literal", "value": "null"}, {"id": 57, "type": "StatementExpression", "children": [58], "value": "priceDate"}, {"id": 58, "type": "Assignment", "children": [59, 60]}, {"id": 59, "type": "MemberReference", "value": "priceDate"}, {"id": 60, "type": "ClassCreator", "children": [61, 62]}, {"id": 61, "type": "ReferenceType", "value": "Timestamp"}, {"id": 62, "type": "MethodInvocation", "value": "System.currentTimeMillis"}, {"id": 63, "type": "StatementExpression", "children": [64], "value": "log.config"}, {"id": 64, "type": "MethodInvocation", "children": [65], "value": "log.config"}, {"id": 65, "type": "BinaryOperation", "children": [66, 71]}, {"id": 66, "type": "BinaryOperation", "children": [67, 70]}, {"id": 67, "type": "BinaryOperation", "children": [68, 69]}, {"id": 68, "type": "Literal", "value": "\"M_PriceList_ID=\""}, {"id": 69, "type": "MemberReference", "value": "M_PriceList_ID"}, {"id": 70, "type": "Literal", "value": "\" - \""}, {"id": 71, "type": "MemberReference", "value": "priceDate"}, {"id": 72, "type": "LocalVariableDeclaration", "children": [73, 74], "value": "int"}, {"id": 73, "type": "BasicType", "value": "int"}, {"id": 74, "type": "VariableDeclarator", "children": [75], "value": "retValue"}, {"id": 75, "type": "Literal", "value": "0"}, {"id": 76, "type": "LocalVariableDeclaration", "children": [77, 78], "value": "String"}, {"id": 77, "type": "ReferenceType", "value": "String"}, {"id": 78, "type": "VariableDeclarator", "children": [79], "value": "sql"}, {"id": 79, "type": "BinaryOperation", "children": [80, 89]}, {"id": 80, "type": "BinaryOperation", "children": [81, 88]}, {"id": 81, "type": "BinaryOperation", "children": [82, 87]}, {"id": 82, "type": "BinaryOperation", "children": [83, 86]}, {"id": 83, "type": "BinaryOperation", "children": [84, 85]}, {"id": 84, "type": "Literal", "value": "\"SELECT plv.M_PriceList_Version_ID, plv.ValidFrom \""}, {"id": 85, "type": "Literal", "value": "\"FROM M_PriceList pl, M_PriceList_Version plv \""}, {"id": 86, "type": "Literal", "value": "\"WHERE pl.M_PriceList_ID=plv.M_PriceList_ID\""}, {"id": 87, "type": "Literal", "value": "\" AND plv.IsActive='Y'\""}, {"id": 88, "type": "Literal", "value": "\" AND pl.M_PriceList_ID=? \""}, {"id": 89, "type": "Literal", "value": "\"ORDER BY plv.ValidFrom DESC\""}, {"id": 90, "type": "TryStatement", "children": [91, 97, 101, 105, 125, 127, 129], "value": "try"}, {"id": 91, "type": "LocalVariableDeclaration", "children": [92, 93], "value": "PreparedStatement"}, {"id": 92, "type": "ReferenceType", "value": "PreparedStatement"}, {"id": 93, "type": "VariableDeclarator", "children": [94], "value": "pstmt"}, {"id": 94, "type": "MethodInvocation", "children": [95, 96], "value": "DB.prepareStatement"}, {"id": 95, "type": "MemberReference", "value": "sql"}, {"id": 96, "type": "Literal", "value": "null"}, {"id": 97, "type": "StatementExpression", "children": [98], "value": "pstmt.setInt"}, {"id": 98, "type": "MethodInvocation", "children": [99, 100], "value": "pstmt.setInt"}, {"id": 99, "type": "Literal", "value": "1"}, {"id": 100, "type": "MemberReference", "value": "M_PriceList_ID"}, {"id": 101, "type": "LocalVariableDeclaration", "children": [102, 103], "value": "ResultSet"}, {"id": 102, "type": "ReferenceType", "value": "ResultSet"}, {"id": 103, "type": "VariableDeclarator", "children": [104], "value": "rs"}, {"id": 104, "type": "MethodInvocation", "value": "pstmt.executeQuery"}, {"id": 105, "type": "WhileStatement", "children": [106, 111], "value": "while"}, {"id": 106, "type": "BinaryOperation", "children": [107, 108]}, {"id": 107, "type": "MethodInvocation", "value": "rs.next"}, {"id": 108, "type": "BinaryOperation", "children": [109, 110]}, {"id": 109, "type": "MemberReference", "value": "retValue"}, {"id": 110, "type": "Literal", "value": "0"}, {"id": 111, "type": "BlockStatement", "children": [112, 117], "value": "{"}, {"id": 112, "type": "LocalVariableDeclaration", "children": [113, 114], "value": "Timestamp"}, {"id": 113, "type": "ReferenceType", "value": "Timestamp"}, {"id": 114, "type": "VariableDeclarator", "children": [115], "value": "plDate"}, {"id": 115, "type": "MethodInvocation", "children": [116], "value": "rs.getTimestamp"}, {"id": 116, "type": "Literal", "value": "2"}, {"id": 117, "type": "IfStatement", "children": [118, 120], "value": "if"}, {"id": 118, "type": "MethodInvocation", "children": [119], "value": "priceDate.before"}, {"id": 119, "type": "MemberReference", "value": "plDate"}, {"id": 120, "type": "StatementExpression", "children": [121], "value": "retValue"}, {"id": 121, "type": "Assignment", "children": [122, 123]}, {"id": 122, "type": "MemberReference", "value": "retValue"}, {"id": 123, "type": "MethodInvocation", "children": [124], "value": "rs.getInt"}, {"id": 124, "type": "Literal", "value": "1"}, {"id": 125, "type": "StatementExpression", "children": [126], "value": "rs.close"}, {"id": 126, "type": "MethodInvocation", "value": "rs.close"}, {"id": 127, "type": "StatementExpression", "children": [128], "value": "pstmt.close"}, {"id": 128, "type": "MethodInvocation", "value": "pstmt.close"}, {"id": 129, "type": "CatchClause", "children": [130, 131]}, {"id": 130, "type": "CatchClauseParameter", "value": "e"}, {"id": 131, "type": "StatementExpression", "children": [132], "value": "log.log"}, {"id": 132, "type": "MethodInvocation", "children": [133, 134, 135], "value": "log.log"}, {"id": 133, "type": "MemberReference", "value": "Level.SEVERE"}, {"id": 134, "type": "MemberReference", "value": "sql"}, {"id": 135, "type": "MemberReference", "value": "e"}, {"id": 136, "type": "StatementExpression", "children": [137], "value": "Env.setContext"}, {"id": 137, "type": "MethodInvocation", "children": [138, 139, 140, 141], "value": "Env.setContext"}, {"id": 138, "type": "MethodInvocation", "value": "Env.getCtx"}, {"id": 139, "type": "MemberReference", "value": "p_WindowNo"}, {"id": 140, "type": "Literal", "value": "\"M_PriceList_Version_ID\""}, {"id": 141, "type": "MemberReference", "value": "retValue"}, {"id": 142, "type": "ReturnStatement", "children": [143], "value": "return"}, {"id": 143, "type": "MemberReference", "value": "retValue"}]
# build_input(ast_example)


f = open("../../data/java/ast.json","r")
train_num = 69708
dev_num, test_num = 8714, 8714
asts = f.readlines()
f1 = open("../../data/java/train3.token.code","w",encoding="utf-8")
f2 = open("../../data/java/valid3.token.code","w",encoding="utf-8")
f3 = open("../../data/java/test3.token.code","w",encoding="utf-8")
for i, ast in tqdm(enumerate(asts)):
    ast = json.loads(ast)
    ast_token,att_matrix = build_input(ast)
    # np.save(f"../../data/java/att/{i}.npy",att_matrix)
    np.savez_compressed(f"../../data/java/att3/{i}.npz", data=att_matrix.data, row=att_matrix.row, col=att_matrix.col,
                        shape=att_matrix.shape)

    if i<train_num:
        f1.write(' '.join(ast_token)+'\n')
    elif i<train_num+dev_num:
        f2.write(' '.join(ast_token)+'\n')
    elif i<train_num + dev_num + test_num:
        f3.write(' '.join(ast_token)+'\n')
