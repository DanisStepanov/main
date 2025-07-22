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
        metrics_df = pd.read_csv('metrics_okr.csv')
        return metrics_df.set_index('metric')['value'].to_dict()


    df = load_data()

    model = load_model()

    priznak = [
    'sqft_living_scaled', 'bedrooms', 'bathrooms', 
    'grade_scaled', 'view_scaled', 'floors',
    'waterfront', 'yr_built','sqft_above_scaled','condition_scaled', 'sqft_basement_scaled', 'yr_renovated','lat','long','sqft_lot_scaled']

    X = df[priznak]
    y = df['price']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    y_pred = model.predict(X_test)


    st.title("Степанов Денис Алексеевич_2023-ФГиИБ-ПИ-1б_20 Вариант_Рынок недвижемости")
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
        В своей работе я рассматривал рынок недвижимости USA до 2024 года.

        **id** — Уникальный идентификационный номер, присвоенный каждому дому в наборе данных.  
        **date** — Дата добавления дома в набор данных в формате ГГГГ-ММ-ДД.  
        **price** — Цена дома в долларах США.  
        **bedrooms** — Количество спален в доме, в наборе данных встречаются дома с 0 до 33 спален.  
        **bathrooms** — Количество ванных комнат в доме, варьируется от 0 до 8.  
        **sqft_living** — Площадь жилой зоны в квадратных футах.  
        **sqft_lot** — Общая площадь участка в квадратных футах.  
        **floors** — Количество этажей в доме.  
        **waterfront** — Индикатор расположения дома у воды (озеро или пляж): 0 — нет, 1 — да.  
        **view** — Оценка вида на город, озеро или пляж из дома, от 0 до 5.  """)
    with col2:
        st.markdown("""
                    

        **condition** — Общая оценка состояния дома, от 1 до 5.  
        **grade** — Общая оценка качества дома, от 1 до 12.  
        **sqft_above** — Площадь дома над уровнем земли в квадратных футах.  
        **sqft_basement** — Площадь подвала дома (ниже уровня земли) в квадратных футах.  
        **yr_built** — Год постройки дома.  
        **yr_renovated** — Год проведения ремонта или реконструкции дома.  
        **zipcode** — Почтовый индекс (5 цифр), в котором расположен дом.  
        **lat** — Географическая широта расположения дома.  
        **long** — Географическая долгота расположения дома.  
        **sqft_living15** — Средняя площадь жилой зоны 15 ближайших домов в квадратных футах.  
        **sqft_lot15** — Средняя площадь участка 15 ближайших домов в квадратных футах.
        """)

    
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["Графики зависимостей","SHAP анализ и результаты обучения модели","Топ прогнозов по ошибке и распределение цен", "Метрики модели", "Исходные данные"])
    
    with tab1:
            st.subheader("Зависимость цены от характеристик дома")
                
            col1, col2 = st.columns(2)
                
            with col1:
                chart1 = alt.Chart(df).mark_boxplot(extent='min-max').encode(
                x=alt.X('bedrooms:O', title='Количество спален', axis=alt.Axis(labelAngle=0)),
                y=alt.Y('price:Q', title='Цена', scale=alt.Scale(zero=False))
            ).properties(
                width=400,
                height=300,
                title="Распределение цен по количеству спален"
            )
            
            mean_line = alt.Chart(df).mark_point(filled=True, size=50, color='red').encode(
                x='bedrooms:O',
                y='mean(price):Q'
            )
            
            st.altair_chart(chart1 + mean_line, use_container_width=True)
            st.write('''
            Наибольший разброс цен у домов с 3-4 спальнями.
            Средняя цена растет от 1 до 4 спален, затем не изменяется существенно, потом резкий рост на 9 спален. После 9 спален средняя цена падает вниз.
            Красные точки показывают средние значения
            ''')

            chart5 = alt.Chart(df).mark_circle(size=8, opacity=0.6).encode(
                x=alt.X('lat:Q', title='Широта', bin=alt.Bin(maxbins=30)),
                y=alt.Y('price:Q', title='Цена', scale=alt.Scale(domain=[0, 2000000])),
                color=alt.Color('mean(price):Q', scale=alt.Scale(scheme='goldorange')),
                size='count()',
                tooltip=['lat', 'mean(price)', 'count()']
            ).properties(
                width=400,
                height=300,
                title="Географическое распределение цен (широта)"
            ).interactive()
            
            st.altair_chart(chart5, use_container_width=True)
            st.write('''
            Самые дорогие дома находятся 47.5-47.7 широты. Чем краснее цвет круга, тем дороже дом.
            ''')

            with col2:
                chart2 = alt.Chart(df).mark_circle(size=40, opacity=0.5).encode(
                x=alt.X('bathrooms:Q', 
                    title='Количество ванных комнат',
                    scale=alt.Scale(nice=False)),
                y=alt.Y('price:Q', title='Цена'),
                tooltip=['bathrooms', 'price']
            ).properties(
                width=400,
                height=300,
                title="Зависимость цены от количества ванных комнат"
            )
            
            trend = chart2.transform_regression(
                'bathrooms', 'price'
            ).mark_line(color='red')
            
            st.altair_chart(chart2 + trend, use_container_width=True)
            st.write('''
            Красная линия тренда, уверенно движется вверх, то есть чем больше ванных комнат, тем выше цена.
            ''')

            chart6 = alt.Chart(df).mark_rect().encode(
                x=alt.X('long:Q', title='Долгота', bin=alt.Bin(maxbins=30)),
                y=alt.Y('price:Q', title='Цена', bin=alt.Bin(maxbins=20, extent=[0, 2000000])),
                color=alt.Color('mean(price):Q', 
                            scale=alt.Scale(scheme='greenblue'),
                            legend=alt.Legend(title="Средняя цена"))
            ).properties(
                width=400,
                height=300,
                title="Распределение цен по долготе"
            )
            
            st.altair_chart(chart6, use_container_width=True)
            st.write('''
            Самые дорогие дома находтся на долготе -122.45 до -122. Чем синее цветы тем дороже, чем белее тем дешевле дома.
            ''')
            # chart4 = alt.Chart(df).mark_bar().encode(
            #     x=alt.X('floors:O', title='Количество этажей'),
            #     y=alt.Y('mean(price):Q', title='Цена'),
            #     tooltip=['floors', 'price']
            # ).properties(
            #     width=400,
            #     height=300,
            #     title="Цена по количеству этажей"
            # )
            # st.altair_chart(chart4, use_container_width=True)
            # st.write('Связь между площадью участка  и ценой. Резкий рост цены , затем слабая зависимость, имеются отдельные участки с большой площадью, скорее всего особняки, но из-за маленькой цены, видимо плохого качества.Площадь была стандартизирована')


    with tab2:
        col1, col2 = st.columns(2)
        
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


    with tab3:
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
        st.write('График,показывающий, как часто встречаются дома в разных ценовых диапазонах. Пик в области $300–500k — большинство домов среднего класса.Длинный хвост справа — редкие дорогие объекты (выбросы). Вывод: Данные смещены в сторону недорогих домов.')

        
            
    with tab4:
        st.subheader("Оценка качества модели")
        col1, col2, col3, col4, col5 = st.columns(5)

        metric = load_metrics()

        r2_value = metric.get('R2', metric.get('R²', 0))
        mae_value = metric.get('MAE', 0)
        max_error_value = metric.get('Max Error', 0)
        med_abs_proc_error = metric.get('Медианная абсолютная процентная ошибка', 0)
        MSE = metric.get('MSE', 0)

        col1.metric("R² (Коэф. детерминации)", f"{r2_value:.2f}")
        col2.metric("MAE (Средняя ошибка)", f"{mae_value:,.0f}")
        col3.metric("Max Error", f"{max_error_value:,.0f}")
        col4.metric("Медианная абсолютная процентная ошибка", f"{med_abs_proc_error:,.1f}%")
        col5.metric("MSE", f"{MSE:,.0f}")

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

