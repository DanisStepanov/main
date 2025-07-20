import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
from sklearn.model_selection import train_test_split
import shap
import matplotlib.pyplot as plt
from xgboost import XGBRegressor
import xgboost as xgb

def main():
    st.set_page_config(layout="wide", page_title="Анализ цен на недвижимость")
    
    @st.cache_data
    def load_data():
        return pd.read_csv('table2.csv')

    def load_model():
        model = XGBRegressor()
        model.load_model('model.json')
        return model
    
    
    def load_metrics():
        metrics_df = pd.read_csv('metrics.csv')
        return metrics_df.set_index('metric')['value'].to_dict()


    df = load_data()

    model = load_model()

    priznak = [
    'sqft_living_scaled', 'bedrooms', 'bathrooms', 'sqft_lot_scaled', 
    'grade_scaled', 'view_scaled', 'floors','sqft_above_scaled',
    'waterfront', 'yr_built','sqft_above','condition_scaled', 'sqft_basement', 'yr_renovated','lat','long',]

    X = df[priznak]
    y = df['price']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    y_pred = model.predict(X_test)


    st.title("Степанов Денис Алексеевич_2023-ФГиИБ-ПИ-1б_20 Вариант_Рынок недвижемости")
    st.write("В своей работе я рассматривал рынок недвижимости до 2024 года.id — Уникальный идентификационный номер, присвоенный каждому дому в наборе данных." \
    " date — Дата добавления дома в набор данных в формате ГГГГ-ММ-ДД."\
    " price — Цена дома в долларах США. bedrooms — Количество спален в доме, в наборе данных встречаются дома с 0 до 33 спален."\
    " bathrooms — Количество ванных комнат в доме, варьируется от 0 до 8. sqft_living — Площадь жилой зоны в квадратных футах. "\
    "sqft_lot — Общая площадь участка в квадратных футах. floors — Количество этажей в доме."\
    " waterfront — Индикатор расположения дома у воды (озеро или пляж): 0 — нет, 1 — да." \
    " view — Оценка вида на город, озеро или пляж из дома, от 0 до 5. condition — Общая оценка состояния дома, от 1 до 5." \
    " grade — Общая оценка качества дома, от 1 до 12. sqft_above — Площадь дома над уровнем земли в квадратных футах." \
    " sqft_basement — Площадь подвала дома (ниже уровня земли) в квадратных футах. yr_built — Год постройки дома." \
    " yr_renovated — Год проведения ремонта или реконструкции дома. zipcode — Почтовый индекс (5 цифр), в котором расположен дом." \
    " lat — Географическая широта расположения дома. long — Географическая долгота расположения дома. " \
    "sqft_living15 — Средняя площадь жилой зоны 15 ближайших домов в квадратных футах." \
    " sqft_lot15 — Средняя площадь участка 15 ближайших домов в квадратных футах.")
    
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["SHAP анализ и результаты обучения модели","Топ прогнозов по ошибке и распределение цен","Графики зависимостей", "Метрики модели", "Исходные данные"])
    
    with tab1:
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.subheader("Важность признаков")
            fig, ax = plt.subplots(figsize=(6, 4))
            xgb.plot_importance(model, ax=ax, height=0.5)
            plt.tight_layout()
            st.pyplot(fig)
            st.write('На этом графике показаны ключевые признаки, которые сильнее всего влияют на предсказание цены дома. Чем длиннее столбец, тем важнее признак для модели.')
        
        with col2:
            st.subheader("SHAP-анализ")
            explainer = shap.Explainer(model)
            shap_values = explainer(X_test)
            fig2 = plt.figure(figsize=(6, 4))
            shap.plots.beeswarm(shap_values, max_display=20, show=False)
            plt.tight_layout()
            st.pyplot(fig2)
            st.write('График показывает, как каждый признак (по оси Y) влияет на прогноз: Красные точки — высокие значения признака увеличивают цену. Синие точки — низкие значения уменьшают цену.')
        with col3:
            st.subheader("Сравнение фактических и предсказанных цен")
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.scatter(y_test, y_pred, alpha=0.5, label="Прогнозы")
            lims = [
            np.min([y_test.min(), y_pred.min()]),
            np.max([y_test.max(), y_pred.max()]),
            ]
            ax.plot(lims, lims, 'r--', label="Идеальная линия")
            ax.set_xlabel("Фактическая цена ($)", fontsize=12)
            ax.set_ylabel("Предсказанная цена ($)", fontsize=12)
            ax.legend()
            ax.grid(True)
            st.pyplot(fig)
            st.write('Точечный график, где по оси X — реальные цены, по оси Y — предсказанные. Идеальная модель даёт точки на красной линии (y = x). Исходя из графика, можно скачазать, что большинство всё было правильно.')
    with tab2:
        st.subheader("Топ прогнозов по ошибке")
        results_df = pd.DataFrame({
            'Факт': y_test,
            'Прогноз': y_pred,
            'Ошибка ($)': abs(y_test - y_pred),
            'Ошибка (%)': abs(y_test - y_pred) / y_test * 100
        }).sort_values('Ошибка ($)', ascending=False)
        st.write('Список домов, где модель ошиблась сильнее всего. В нашем случае более 20 000 домов и средняя процентная ошибка 12%, это хороший результат.')
        
        st.dataframe(
            results_df.head(100).style.format({
                'Факт': '${:,.0f}',
                'Прогноз': '${:,.0f}',
                'Ошибка ($)': '${:,.0f}',
                'Ошибка (%)': '{:.1f}%'
            }),
            height=300,
            use_container_width=True
        )

        st.header("2. Распределение цен")
        chart = alt.Chart(df).mark_bar().encode(
            alt.X("price:Q", bin=True, title="Цена"),
            alt.Y("count()", title="Количество домов")
        )
        st.altair_chart(chart, use_container_width=True)
        st.write('График,показывающий, как часто встречаются дома в разных ценовых диапазонах. Пик в области $300–500k — большинство домов среднего класса.Длинный хвост справа — редкие дорогие объекты (выбросы). Вывод: Данные смещены в сторону недорогих домов, что может влиять на ошибки для luxury-сегмента.')

    with tab3:
        st.subheader("Зависимость цены от характеристик дома")
        
        col1, col2 = st.columns(2)
        
        with col1:
            chart1 = alt.Chart(df).mark_circle().encode(
                x=alt.X('bedrooms:O', axis=alt.Axis(title='Количество спален')),
                y=alt.Y('price:Q', axis=alt.Axis(title='Цена')),
                tooltip=['bedrooms', 'price']
            ).properties(
                width=400,
                height=300,
                title="Цена от количества спален"
            ).interactive()
            st.altair_chart(chart1, use_container_width=True)
            st.write('Показывает, как количество комнат влияет на стоимость дома. Точки определяют каждый отдельный дом, самое популярное количество домов с 2-5 ванными.')
            
            chart3 = alt.Chart(df).mark_circle().encode(
                x=alt.X('sqft_living:Q', axis=alt.Axis(title='Жилая площадь (кв. футы)')),
                y=alt.Y('price:Q', axis=alt.Axis(title='Цена')),
                tooltip=['sqft_living', 'price']
            ).properties(
                width=400,
                height=300,
                title="Цена от жилой площади"
            ).interactive()
            st.altair_chart(chart3, use_container_width=True)
            st.write('Связь между жилой площадью участка (в квадратных футах) и ценой. Рост наблюдается на всём графике.')
            
            chart5 = alt.Chart(df).mark_circle().encode(
                x=alt.X('view_scaled:Q', axis=alt.Axis(title='Оценка вида')),
                y=alt.Y('price:Q', axis=alt.Axis(title='Цена')),
                tooltip=['view_scaled', 'price']
            ).properties(
                width=400,
                height=300,
                title="Цена от оценки вида"
            ).interactive()
            st.altair_chart(chart5, use_container_width=True)
            st.write('Зависимость цены от экспертной оценки вида из дома (шкала 0-1). Чем выше оценка, тем выше цена. Почти линейная зависимость.')
        
        with col2:
            chart2 = alt.Chart(df).mark_circle().encode(
                x=alt.X('bathrooms:O', axis=alt.Axis(title='Количество ванных комнат')),
                y=alt.Y('price:Q', axis=alt.Axis(title='Цена')),
                tooltip=['bathrooms', 'price']
            ).properties(
                width=400,
                height=300,
                title="Цена от количества ванных"
            ).interactive()
            st.altair_chart(chart2, use_container_width=True)
            st.write('Показывает, как количество ванных комнат влияет на стоимость дома. Рост цены при увеличении ванных до определенного предела (3-3,5).')

            chart4 = alt.Chart(df).mark_circle().encode(
                x=alt.X('sqft_lot_scaled:Q', axis=alt.Axis(title='Площадь участка.')),
                y=alt.Y('price:Q', axis=alt.Axis(title='Цена')),
                tooltip=['sqft_lot_scaled', 'price']
            ).properties(
                width=400,
                height=300,
                title="Цена от площади участка"
            ).interactive()
            st.altair_chart(chart4, use_container_width=True)
            st.write('Связь между площадью участка  и ценой. Резкий рост цены , затем слабая зависимость, имеются отдельные участки с большой площадью, скорее всего особняки, но из-за маленькой цены, видимо плохого качества.Площадь была стандартизирована')

            chart6 = alt.Chart(df).mark_circle().encode(
                x=alt.X('grade_scaled:Q', axis=alt.Axis(title='Оценка качества')),
                y=alt.Y('price:Q', axis=alt.Axis(title='Цена')),
                tooltip=['grade_scaled', 'price']
            ).properties(
                width=400,
                height=300,
                title="Цена от оценки качества"
            ).interactive()
            st.altair_chart(chart6, use_container_width=True)
            st.write('Зависимость цены от экспертной оценки качества дома (шкала 0-1). Чем выше оценка, тем выше цена. Почти линейная зависимость.')

        
            
    with tab4:
        st.subheader("Оценка качества модели")
        col1, col2, col3, col4, col5 = st.columns(5)

        metric = load_metrics()

        r2_value = metric.get('R2', metric.get('R²', 0))
        mae_value = metric.get('MAE', 0)
        max_error_value = metric.get('Max Error', 0)
        med_abs_proc_error = metric.get('Медианная абсолютная процентная ошибка', 0)
        MSE = metric.get('MSE', 0)

        col1.metric("R² (Коэф. детерминации)", f"{r2_value:.3f}")
        col2.metric("MAE (Средняя ошибка)", f"{mae_value:,.0f}")
        col3.metric("Max Error", f"{max_error_value:,.0f}")
        col4.metric("Медианная абсолютная процентная ошибка", f"%{med_abs_proc_error:,.3f}")
        col5.metric("MSE", f"{MSE:,.3f}")

        st.subheader("Матрица корреляции признаков")
        corr = X_train.corr().stack().reset_index(name='correlation')
        chart_corr = alt.Chart(corr).mark_rect().encode(
            x=alt.X('level_0:N', axis=alt.Axis(title='Признак 1')),
            y=alt.Y('level_1:N', axis=alt.Axis(title='Признак 2')),
            color='correlation:Q',
            tooltip=['correlation']
        ).properties(
            width=600,
            height=500
        ).interactive()
        st.altair_chart(chart_corr, use_container_width=True)
        
    with tab5:
        st.subheader("Исходные данные")
        st.dataframe(df, use_container_width=True)


if __name__ == "__main__":
    main()

