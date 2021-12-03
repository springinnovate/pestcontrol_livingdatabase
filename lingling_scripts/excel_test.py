#导入库
import  pandas  as pd
import xlwt

def excel_test():
    #传入excel文件名，最好是.xls格式。 与sheet名
    df=pd.read_excel('G:/Python/excel/test_Landsat7_EVI_1999_2020_on_pixel.xlsx',sheet_name='Landsat7_EVI_1999_2020_on_pixel')

    #获取数据的第一行，即索引行
    statistic_name = list(df)
    #获取数据的最后一列，即年份列
    years = list(df[df.columns[-1]])

    #创建空列表，作为结果存储
    s_max = []
    s_min = []
    s_mean = []
    s_stdDev = []

    #循环索引行与年份列
    for i in range(len(years)):
        for j in range(len(statistic_name)):
            #判断年份列是否在索引行中
            if str(years[i]) in str(statistic_name[j]):
                #若上一判断满足，判断字符串'max'在索引行中，以下同理。
                if 'max' in str(statistic_name[j]):
                    s_max.append(df.iat[i, j])
                if 'min' in str(statistic_name[j]):
                    s_min.append(df.iat[i, j])
                if 'mean' in str(statistic_name[j]):
                    s_mean.append(df.iat[i, j])
                if 'stdDev' in str(statistic_name[j]):
                    s_stdDev.append(df.iat[i, j])
    #输出max、min、mean、stdDev的元素数目，看是否相同。
    print(len(s_max), len(s_min), len(s_mean), len(s_stdDev))

    #创建字典，把上述4个列表合并。
    statistic_result = {
        "max": s_max,
        "min": s_min,
        "mean": s_mean,
        "stdDev": s_stdDev
    }
    #将字典转为Dataframe，便于excel导出。
    statistic_result = pd.DataFrame(statistic_result)

    #取原数据的后10列，即点位信息。
    point_info = df.iloc[:, -10:]

    #将上述筛选结果与点位信息的Dataframe合并
    result = pd.concat([statistic_result, point_info], axis=1)

    #输出为excel
    result.to_excel('test_Landsat7_EVI_1999_2020_on_pixel_result.xls')


if __name__ == '__main__':
    excel_test()

