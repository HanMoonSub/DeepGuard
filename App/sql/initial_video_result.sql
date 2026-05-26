drop table if exists video_result;

-- 딥페이크 분석 결과를 저장하는 테이블 생성
create table deepfake_db.video_result (
    id integer auto_increment primary key, -- 시스템에서 부여하는 데이터 고유 식별 번호
    user_id integer null,                  -- 영상을 업로드한 사용자의 ID (비회원 허용을 위해 null 가능)
    video_loc varchar(300) not null unique, -- 영상 파일이 저장된 경로/URL (중복 등록 방지)
    
    -- 상태 및 결과 정보
    status varchar(20) not null default "PENDING", -- 분석 진행 상태 (대기중, 분석중, 완료, 실패 등)
    label varchar(10) not null,                    -- 최종 판정 결과 (예: 'Fake', 'Real')
    score float not null, 
    
    -- 분석 품질 및 메타데이터
    face_conf float not null,
    face_ratio float not null,
    face_brightness float not null,
    
    -- 분석 환경 정보
    version_type varchar(10) not null,
    model_type varchar(10) not null,
    domain_type varchar(20) not null,
    
    -- 기타 기록
    result_msg varchar(200) not null,  -- 분석 결과에 대한 상세 메시지나 오류 내용
    created_at timestamp default current_timestamp, -- 데이터가 생성된(분석 요청된) 시간
    
    -- 성능 최적화
    index user_id_idx(user_id) -- 특정 사용자의 분석 이력을 빠르게 조회하기 위한 인덱스
);