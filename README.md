# Regressao_Polinomial

A regressão polinomial funciona transformando as variáveis originais em novas variáveis que são potências dessas variáveis.

Por exemplo, se originalmente temos apenas a feature X, podemos construir novas features como X^2, X^3, X^4, etc. Então encaixamos um modelo linear nessas novas variáveis.

Matematicamente, a regressão polinomial assume esta forma:

Y = a + b1*X + b2*X^2 + ... + bp*X^p


Onde p é o grau do polinômio. Ao incluir os termos de potência superiores, podemos modelar curvaturas e tendências não lineares em Y.

Por exemplo, o gráfico abaixo mostra um modelo quadrático (grau 2) encaixando bem uma relação não linear:

Regressão Polinomial

A grande vantagem da regressão polinomial é sua simplicidade. Não precisamos fazer muitas transformações complexas nos dados. Basta computar potências superiores de X e encaixar um modelo linear padrão.

No Python, isso é fácil com scikit-learn:

from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

poly = PolynomialFeatures(degree=2)
X_poly = poly.fit_transform(X)

modelo = LinearRegression()
modelo.fit(X_poly, y)


O PolynomialFeatures cria as novas features com os termos de potência.

Um ponto de atenção com a regressão polinomial é não exagerar no grau do polinômio. Polinômios de grau muito alto tendem a overfit nos dados. Normalmente graus 2 ou 3 já são suficientes para uma boa aproximação.

Comparando as Técnicas

Vimos três poderosas técnicas para contornar a limitação da linearidade dos modelos lineares:


Transformações log: Úteis para relações exponenciais/potência
Splines: Excelentes para relações arbitrariamente complexas
Regressão Polinomial: Simples de implementar e computacionalmente eficiente

Qual técnica usar vai depender muito dos seus dados e do tipo de relação que você precisa modelar.

De modo geral, eu recomendo começar testando com regressão polinomial, por ser mais simples. Se o polinômio não der um bom ajuste, experimente splines ou logs.

Também é possível combinar técnicas, por exemplo, usando splines e depois ajustando um polinômio nos splines. As possibilidades são infinitas!

O importante é sempre verificar se as transformações estão melhorando o ajuste do modelo aos dados (R^2, erro absoluto médio etc). Transformações mais complexas não necessariamente resultam em melhorias.

Conclusão

A não linearidade representa um desafio para modelos de machine learning baseados em regressão linear. Felizmente, com algumas transformações criativas nas variáveis, podemos contornar em grande parte esse problema e ainda assim desfrutar da simplicidade e eficiência dos modelos lineares.

Técnicas como regressão polinomial, splines e transformações logarítmicas são ferramentas poderosas que devemos ter no nosso arsenal. Quando aplicadas corretamente, elas podem melhorar muito as métricas e capacidade preditiva dos modelos.
