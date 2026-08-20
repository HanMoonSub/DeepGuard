FROM mysql:26.7-oraclelinux9

COPY App/sql/*.sql /docker-entrypoint-initdb.d/

COPY docker/mysql/config/mysql.cnf /etc/mysql/conf.d/mysql.cnf

ENV LANG=C.UTF-8
ENV TZ=Asia/Seoul

EXPOSE 3306