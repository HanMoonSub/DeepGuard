drop table if exists image_result;

CREATE TABLE image_result (
    id           INT AUTO_INCREMENT PRIMARY KEY,
    user_id      INT NOT NULL,
    image_loc    VARCHAR(300) NOT NULL UNIQUE,
    label        TINYINT NOT NULL,
    score        FLOAT NOT NULL,
    version_type VARCHAR(10) NOT NULL,
    model_type   VARCHAR(10) NOT NULL,
    domain_type  VARCHAR(20) NOT NULL,
    created_at   TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    
    INDEX user_id_idx(user_id)
);