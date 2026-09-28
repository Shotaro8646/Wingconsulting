-- BorderFlow AI の保存先。StayPath とは別テーブルです。
-- Supabase の SQL Editor で、このファイルをそのまま実行してください。
-- anon key を知っている人はこの1テーブルだけ読み書きできます。他のテーブルは開きません。

create table if not exists public.borderflow_workspace (
  id text primary key,
  deals jsonb not null default '[]'::jsonb,
  customers jsonb not null default '[]'::jsonb,
  voyages jsonb not null default '[]'::jsonb,
  updated_at timestamptz not null default now()
);

alter table public.borderflow_workspace enable row level security;

grant select, insert, update on public.borderflow_workspace to anon, authenticated;

drop policy if exists borderflow_workspace_read on public.borderflow_workspace;
drop policy if exists borderflow_workspace_insert on public.borderflow_workspace;
drop policy if exists borderflow_workspace_update on public.borderflow_workspace;

create policy borderflow_workspace_read
  on public.borderflow_workspace
  for select
  to anon, authenticated
  using (true);

create policy borderflow_workspace_insert
  on public.borderflow_workspace
  for insert
  to anon, authenticated
  with check (id = 'default');

create policy borderflow_workspace_update
  on public.borderflow_workspace
  for update
  to anon, authenticated
  using (id = 'default')
  with check (id = 'default');
